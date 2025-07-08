import warnings

from deep_metabolitics.data.metabolight_dataset_pyspark import MultioutDataset
from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset
from deep_metabolitics.networks.pyspark_model import PySparkMultiOutputRegressor

warnings.filterwarnings("ignore")

import os
import random
import joblib


import torch

from deep_metabolitics.config import outputs_dir

from deep_metabolitics.data.properties import get_aycan_dataset_ids

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# from xgboost.spark import XGBRegressor

from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.sql import SparkSession

from deep_metabolitics.utils.performance_metrics import PerformanceMetrics
from deep_metabolitics.utils.trainer_pm import predict_pyspark, train_pyspark
from deep_metabolitics.config import data_dir

single_model_class_list = [
    # RandomForestRegressor,
    GBTRegressor,
    # XGBRegressor,
    # XGBRFRegressor,
    # SVR,
    # MLPRegressor,
]

experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

metabolite_scaler_method = "std"
target_scaler_method = "std"
# metabolite_coverage = "fully"
metabolite_coverage = None
k_folds = 10
batch_size = 32
pathway_features = False
experiment_name = f"{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{k_folds}_{batch_size}"

# ids_list = get_workbench_metabolights_dataset_ids()
# print(f"{len(ids_list) = }")

test_source_list = get_aycan_dataset_ids()

# spark = SparkSession.builder \
#     .appName("BigDataProcessing") \
#     .config("spark.driver.memory", "64g") \
#     .config("spark.executor.memory", "64g") \
#     .config("spark.memory.fraction", "0.5") \
#     .config("spark.memory.storageFraction", "0.3") \
#     .config("spark.sql.shuffle.partitions", "200") \
#     .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
#     .config("spark.sql.files.maxPartitionBytes", "128MB") \
#     .config("spark.local.dir", "./disk_spark") \
#     .getOrCreate()

# spark = SparkSession.builder \
#     .appName("BigDataProcessing") \
#     .master("local[36]") \
#     .config("spark.driver.memory", "180g") \
#     .config("spark.sql.shuffle.partitions", "800") \
#     .config("spark.memory.fraction", "0.8") \
#     .config("spark.memory.storageFraction", "0.3") \
#     .config("spark.driver.maxResultSize", "8g") \
#     .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
#     .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
#     .config("spark.sql.execution.arrow.maxRecordsPerBatch", "100000") \
#     .config("spark.local.dir", "./disk_spark") \
#     .getOrCreate()

spark = SparkSession.builder \
    .appName("BigDataProcessing") \
    .master("local[18]") \
    .config("spark.driver.memory", "180g") \
    .config("spark.sql.shuffle.partitions", "600") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "100000") \
    .config("spark.local.dir", "./disk_spark") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("./spark-checkpoint")

for fold in list(range(k_folds)):
    train_all_dataset = MultioutDataset(
        spark=spark,
        metabolite_fpath=data_dir / "reactionminmax_10_folds" / f"metabolomics_train_{fold}.parquet.gzip",
        label_fpath=data_dir / "reactionminmax_10_folds" / f"label_train_{fold}.parquet.gzip",
        batch_size=batch_size,
        metabolite_coverage=metabolite_coverage,
        pathway_features=pathway_features,
    )
    print(f"train_all_dataset for {fold = } is completed.")
    validation_all_dataset = MultioutDataset(
        spark=spark,
        metabolite_fpath=data_dir / "reactionminmax_10_folds" / f"metabolomics_test_{fold}.parquet.gzip",
        label_fpath=data_dir / "reactionminmax_10_folds" / f"label_test_{fold}.parquet.gzip",
        batch_size=batch_size,
        metabolite_coverage=metabolite_coverage,
        pathway_features=pathway_features,
        metabolite_model=train_all_dataset.metabolite_model,
        label_model=train_all_dataset.label_model,
    )
    print(f"validation_all_dataset for {fold = } is completed.")
    metabolomics_df, label_df, _, _ = ReactionMinMaxDataset.load_data_aycan_csv(dataset_ids=test_source_list)
    test_all_dataset = MultioutDataset(
        spark=spark,
        metabolite_fpath=metabolomics_df,
        label_fpath=label_df,
        batch_size=batch_size,
        metabolite_coverage=metabolite_coverage,
        pathway_features=pathway_features,
        metabolite_model=train_all_dataset.metabolite_model,
        label_model=train_all_dataset.label_model,
    )
    print(f"test_all_dataset for {fold = } is completed.")
    for model_class in single_model_class_list:
        experiment_fold = f"{experiment_name}_{model_class.__name__}_fold_{fold}"

        # scaler = train_all_dataset.scaler
        scaler = None

        n_features = train_all_dataset.n_metabolights
        out_features = train_all_dataset.n_labels

        model = PySparkMultiOutputRegressor(base_model=model_class(), 
                                            feature_columns=train_all_dataset.feature_columns,
                                            target_columns=train_all_dataset.label_names,
                                            label_model=train_all_dataset.label_model)
        model, train_elapsed_time = train_pyspark(
            train_dataset=train_all_dataset, model=model
        )


        joblib.dump(model, outputs_dir / f"{experiment_fold}.joblib")


        pred_train, true_train, _ = predict_pyspark(
            model=model, dataset=train_all_dataset
        )
        pred_validation, true_validation, validation_elapsed_time = predict_pyspark(
            model=model, dataset=validation_all_dataset
        )
        pred_test, true_test, test_elapsed_time = predict_pyspark(
            model=model, dataset=test_all_dataset
        )

        performance_metrics = PerformanceMetrics(
            target_names=list(train_all_dataset.label_names),
            experience_name=experiment_fold,
            train_time=train_elapsed_time,
            test_time=test_elapsed_time,
            validation_time=validation_elapsed_time,
            scaler=scaler,
        )
        performance_metrics.train_metric(y_true=true_train, y_pred=pred_train)
        performance_metrics.validation_metric(
            y_true=true_validation, y_pred=pred_validation
        )
        performance_metrics.test_metric(y_true=true_test, y_pred=pred_test)
        performance_metrics.complete()  # TODO foldlari tek dosyada tutsak guzel olur
