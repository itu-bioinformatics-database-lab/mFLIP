import warnings
import time
from deep_metabolitics.data.oneoutdataset import OneoutDataset
warnings.filterwarnings("ignore")

import os
import random
import joblib

import torch

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    StringIndexer,
    OneHotEncoder
)
from pyspark.sql.functions import lit
# XGBoost Spark integration
# For XGBoost >=1.6 use SparkXGBRegressor, else adjust import accordingly
from xgboost.spark import SparkXGBRegressor



from deep_metabolitics.config import outputs_dir, data_dir

from deep_metabolitics.data.properties import get_aycan_dataset_ids
from deep_metabolitics.data.properties import get_recon_metabolites



seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


spark = SparkSession.builder \
    .appName("BigDataProcessing") \
    .master("local[36]") \
    .config("spark.driver.memory", "180g") \
    .config("spark.sql.shuffle.partitions", "600") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "100000") \
    .config("spark.local.dir", "/tmp/bacan_disk_spark") \
    .getOrCreate()

# .config("spark.local.dir", "./disk_spark") \
# .config("spark.sql.execution.arrow.pyspark.enabled", "true") \


spark.sparkContext.setCheckpointDir("/tmp/bacan_spark-checkpoint")
# spark.sparkContext.setCheckpointDir("./spark-checkpoint")
spark.conf.set('spark.sql.caseSensitive', True)
# spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")





experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

metabolite_scaler_method = "std"
target_scaler_method = None
# metabolite_coverage = "fully"
metabolite_coverage = None
pathway_features = True
k_folds = 10
batch_size = 128

datasource = "pathwayfluxminmax_10_folds"

experiment_name = f"{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{k_folds}_{batch_size}"


# 2. Feature ve label tanımları
categorical_col = "pathway_name"
label_col = "target_value"


all_metabolities = set(get_recon_metabolites())


for fold in list(range(k_folds)):
    train_path = data_dir / datasource / f"train_oneout_{fold}.parquet.gzip"
    validation_path = data_dir / datasource / f"test_oneout_{fold}.parquet.gzip"
    test_path = data_dir / datasource / f"test_oneout_cancer.parquet.gzip"

    train_all_dataset = spark.read.parquet(str(train_path), inferSchema=True)
    validation_all_dataset = spark.read.parquet(str(validation_path), inferSchema=True)
    test_all_dataset = spark.read.parquet(str(test_path), inferSchema=True)
    
    working_metabolites = list(all_metabolities.intersection(train_all_dataset.columns))
    features = working_metabolites + ["bound_avg", "reaction_count"]
    numeric_cols = sorted(features)

    for f in numeric_cols:
        if f not in validation_all_dataset.columns:
            validation_all_dataset = validation_all_dataset.withColumn(f, lit(0.0))

    for f in numeric_cols:
        if f not in test_all_dataset.columns:
            test_all_dataset = test_all_dataset.withColumn(f, lit(0.0))

    train_all_dataset = train_all_dataset.fillna(0)
    validation_all_dataset = validation_all_dataset.fillna(0)
    test_all_dataset = test_all_dataset.fillna(0)


    experiment_fold = f"{experiment_name}_fold_{fold}"
    # train_all_dataset = train_all_dataset.limit(10_000)

    # print(f"{train_all_dataset.columns = }")
    # print(f"{numeric_cols = }")

    missing_cols = [f for f in numeric_cols if f not in train_all_dataset.columns]
    print("Eksik numeric sütunlar:", missing_cols)

    is_missing_cat_col = categorical_col not in train_all_dataset.columns
    print(f"{is_missing_cat_col = }")

    # train_all_dataset.show()

    # 3. Pipeline aşamaları
    # 3.1. Kategorik -> index
    indexer = StringIndexer(
        inputCol=categorical_col,
        outputCol="category_index",
        handleInvalid="keep"
    )
    # 3.2. Index -> one-hot
    encoder = OneHotEncoder(
        inputCol="category_index",
        outputCol="category_ohe",
        handleInvalid="keep"
    )
    # 3.3. Numeric vetorize
    num_assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="numeric_vector"
    )
    # 3.4. Scale numeric vector
    scaler = StandardScaler(
        inputCol="numeric_vector",
        outputCol="numeric_scaled",
        withMean=True,
        withStd=True
    )
    # 3.5. Tüm feature'ları birleştir
    final_assembler = VectorAssembler(
        inputCols=["numeric_scaled", "category_ohe"],
        outputCol="features"
    )
    # 3.6. XGBoost regresyon modeli
    gxgb = SparkXGBRegressor(
        objective="reg:squarederror",
        features_col="features",
        label_col=label_col,
        prediction_col="prediction",
        # örnek parametreler
        # maxDepth=6,
        # eta=0.1,
        # numRound=100,
        # subsample=0.8
    )

    # 4. Pipeline oluştur
    pipeline = Pipeline(
        stages=[
            indexer,
            encoder,
            num_assembler,
            scaler,
            final_assembler,
            gxgb
        ]
    )

    # 5. Modeli fit et
    # spark_df: önceden yüklenmiş DataFrame
    start_time = time.time()
    model = pipeline.fit(train_all_dataset)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Train fit {elapsed_time = }")

    # 6. Tahmin
    start_time = time.time()
    fpath = os.path.join(
        outputs_dir, f"predictions_train_{experiment_fold}.parquet.gzip"
    )
    predictions = model.transform(train_all_dataset)
    predictions.select(["target_name", "target_value", "prediction"]).write.parquet(fpath, compression="gzip", mode="overwrite")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Train transform {elapsed_time = }")

    start_time = time.time()

    fpath = os.path.join(
        outputs_dir, f"predictions_validation_{experiment_fold}.parquet.gzip"
    )
    predictions = model.transform(validation_all_dataset)
    predictions.select(["target_name", "target_value", "prediction"]).write.parquet(fpath, compression="gzip", mode="overwrite")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Validation transform {elapsed_time = }")

    start_time = time.time()

    fpath = os.path.join(
        outputs_dir, f"predictions_test_{experiment_fold}.parquet.gzip"
    )
    predictions = model.transform(test_all_dataset)
    predictions.select(["target_name", "target_value", "prediction"]).write.parquet(fpath, compression="gzip", mode="overwrite")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Test transform {elapsed_time = }")
