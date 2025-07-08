import warnings

from deep_metabolitics.data.metabolight_dataset import PathwayFluxMinMaxDataset

warnings.filterwarnings("ignore")

import os
import gc
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode, array, struct, col, monotonically_increasing_id, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

from deep_metabolitics.data.properties import get_aycan_dataset_ids
from deep_metabolitics.data.properties import get_workbench_metabolights_dataset_ids
from deep_metabolitics.config import data_dir
from deep_metabolitics.utils.utils import (
    load_cobra_network,
)
seed = 10
random.seed(seed)

spark = SparkSession.builder \
    .appName("BigDataProcessing") \
    .master("local[36]") \
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
spark.conf.set('spark.sql.caseSensitive', True)

experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

datasource = "pathwayfluxminmax_10_folds"
out_dir = data_dir / datasource

# metabolite_scaler_method = None
# target_scaler_method = None
# metabolite_coverage = "mm_union"
k_folds = 10
# batch_size = 32


# metabolite_scaler_method="quantile"
# target_scaler_method="autoscaler"
# metabolite_coverage="fully"
# source_list=None
# k_folds=10
filter_ds=[]
# fold_idx=None
test_source_list = get_aycan_dataset_ids()


source_list = get_workbench_metabolights_dataset_ids()
map = {}
fold_temp_datasets = dict()

print(f"{len(source_list) = }")


def get_pathwaybased_mappings():
    mapping = {}
    recon3 = load_cobra_network()

    for reaction in recon3.reactions:
        reaction_name = reaction.id
        lower_bound, upper_bound = reaction.bounds
        pathway_name = reaction.subsystem
        if pathway_name not in mapping:
            mapping[f"{pathway_name}_min"] = [0, 0, pathway_name]
            mapping[f"{pathway_name}_max"] = [0, 0, pathway_name]
        mapping[f"{pathway_name}_min"][0] += lower_bound
        mapping[f"{pathway_name}_max"][0] += upper_bound
        mapping[f"{pathway_name}_min"][1] += 1
        mapping[f"{pathway_name}_max"][1] += 1



        for pathway_name in mapping:
            mapping[pathway_name][0] /= mapping[pathway_name][1]
    del recon3
    gc.collect()
    return mapping

mapping = get_pathwaybased_mappings()
def repeat_dataset(X: pd.DataFrame, y: pd.DataFrame):
    # Feature'ları target sayısı kadar çoğalt
    X_repeat = pd.concat([X] * y.shape[1], ignore_index=True)

    # Target'ları uzun formata getir
    y_melt = y.melt(var_name="target_name", value_name="target_value")

    # Birleştir
    final_df = pd.concat([X_repeat, y_melt], axis=1)
    

    final_df[['bound_avg', 'reaction_count', 'pathway_name']] = final_df['target_name'].map(mapping).apply(pd.Series)
    return final_df


def generate_stack_expr(columns):
    expr = "stack({}, {})".format(
        len(columns),
        ",".join([
            f"'{col}', `{col}`" for col in columns
        ])
    )
    return expr


def repeat_dataset_spark(X_spark, y_spark, mapping_dict):
    from pyspark.sql.functions import monotonically_increasing_id
    # window = Window.orderBy(monotonically_increasing_id())
    # y_spark içindeki tüm target sütun adlarını al
    y_spark = y_spark.drop("__index_level_0__")
    target_cols = y_spark.columns


    # stack fonksiyonu için input string hazırla
    # Örnek: stack(2, 'y1', y1, 'y2', y2)
    # stack_expr = f"stack({len(target_cols)}, " + ", ".join([f"'{col}', `{col}`" for col in target_cols]) + ") as (target_name, target_value)"
    # stack_expr = f"stack({len(target_cols)}, " + ", ".join([f"'{col}', '{col}'" for col in target_cols]) + ") as (target_name, target_value)"
    stack_expr = f"stack({len(target_cols)}, " + ", ".join([f"'{col}', `{col}`" for col in target_cols]) + ") as (target_name, target_value)"


    # y_spark'ı melt/stack et
    melted_y = y_spark.selectExpr(stack_expr)

    # X_spark satırlarını tekrarlamak için cross join yap
    # Ama önce X_spark ve y_spark’a row_id eklemeliyiz ki doğru eşleşsin
    

    print(f"{X_spark.count() = }", f"{y_spark.count() = }", f"{melted_y.count() = }")
    
    
    X_spark.createOrReplaceTempView('X_spark')
    X_with_id = spark.sql('select row_number() over (order by "some_column") as row_id, * from X_spark')

    y_spark.createOrReplaceTempView('y_spark')
    y_with_id = spark.sql('select row_number() over (order by "some_column") as row_id, * from y_spark')
    # X_with_id = X_spark.withColumn("row_id", monotonically_increasing_id())
    # y_with_id = y_spark.withColumn("row_id", monotonically_increasing_id())

    # X_with_id = X_spark.withColumn("row_id",  row_number().over(window))
    # y_with_id = y_spark.withColumn("row_id",  row_number().over(window))

    # X_with_id = X_spark.rdd.zipWithIndex().map(lambda row: Row(**row[0].asDict(), row_id=row[1]))
    # y_with_id = y_spark.rdd.zipWithIndex().map(lambda row: Row(**row[0].asDict(), row_id=row[1]))

    # X_with_id = spark.createDataFrame(X_with_id)
    # y_with_id = spark.createDataFrame(y_with_id)

    # X_with_id = X_spark.rdd.zipWithIndex()
    # y_with_id = y_spark.rdd.zipWithIndex()


    

    # X_with_id.select("row_id").show()
    # y_with_id.select("row_id").show()

    # print(f"{X_with_id.count() = }", f"{y_with_id.count() = }", f"{melted_y.count() = }")


    # melt işlemini yaptıktan sonra row_id ile join
    melted_y = y_with_id.selectExpr("row_id", stack_expr)
    print(f"{X_with_id.count() = }", f"{y_with_id.count() = }", f"{melted_y.count() = }")
    # melted_y.select("row_id").show()

    final_df = X_with_id.join(melted_y, on="row_id", how="inner").drop("row_id")
    # print(f"{final_df.count() = }")


    # 5. mapping sözlüğünü UDF ile kullan
    def map_target_name(name):
        return mapping_dict.get(name, (None, None, None))

    mapping_schema = StructType([
        StructField("bound_avg", FloatType(), True),
        StructField("reaction_count", IntegerType(), True),
        StructField("pathway_name", StringType(), True),
    ])

    mapping_udf = udf(map_target_name, mapping_schema)

    # 6. mapping değerlerini dataframe'e ekle
    final_df = final_df.withColumn("mapping", mapping_udf(col("target_name"))) \
                       .withColumn("bound_avg", col("mapping.bound_avg")) \
                       .withColumn("reaction_count", col("mapping.reaction_count")) \
                       .withColumn("pathway_name", col("mapping.pathway_name")) \
                       .drop("mapping")
    print(f"{final_df.count() = }")

    return final_df


def repeat_dataset_spark_old(X_spark, y_spark, mapping_dict):
    """
    PySpark versiyonu: Feature dataframe'i target sayısı kadar tekrarlar,
    target dataframe'ini uzun formata getirir ve birleştirir.
    mapping_dict ile bound_avg, reaction_count, pathway_name bilgilerini ekler.
    """
    spark = X_spark.sql_ctx.sparkSession  # Spark session'ı al

    # 1. Target'ları uzun forma çevir (melt gibi)
    # stack_expr = generate_stack_expr(y_spark.columns)
    # melted_y = y_spark.selectExpr(
    #     "stack(" + str(len(y_spark.columns)) + ", " +
    #     ",".join([f"'{col}', {col}" for col in y_spark.columns]) +
    #     ") as (target_name, target_value)"
    # )
    # melted_y = y_spark.selectExpr(
    #     "stack(" + str(len(y_spark.columns)) + ", " +
    #     ",".join([f"'{col}', `{col}`" for col in y_spark.columns]) +
    #     ") as (target_name, target_value)"
    # )
    casted_y = y_spark.select([F.col(c).cast("double").alias(c) for c in y_spark.columns])

    melted_y = casted_y.selectExpr(
        "stack(" + str(len(casted_y.columns)) + ", " +
        ",".join([f"'{col}', `{col}`" for col in casted_y.columns]) +
        ") as (target_name, target_value)"
    )
    # 2. X dataframe'ini target sayısı kadar tekrar et
    repeat_count = len(casted_y.columns)
    repeat_df = spark.createDataFrame(range(repeat_count), "int").withColumnRenamed("value", "repeat_index")

    X_repeated = X_spark.withColumn("dummy_id", monotonically_increasing_id()) \
                        .crossJoin(repeat_df) \
                        .withColumn("row_id", monotonically_increasing_id()) \
                        .drop("repeat_index")

    # 3. melted_y’ye id ekle
    melted_y = melted_y.withColumn("row_id", monotonically_increasing_id())

    # 4. Join işlemi
    final_df = X_repeated.join(melted_y, on="row_id").drop("row_id", "dummy_id")

    # 5. mapping sözlüğünü UDF ile kullan
    def map_target_name(name):
        return mapping_dict.get(name, (None, None, None))

    mapping_schema = StructType([
        StructField("bound_avg", FloatType(), True),
        StructField("reaction_count", IntegerType(), True),
        StructField("pathway_name", StringType(), True),
    ])

    mapping_udf = udf(map_target_name, mapping_schema)

    # 6. mapping değerlerini dataframe'e ekle
    final_df = final_df.withColumn("mapping", mapping_udf(col("target_name"))) \
                       .withColumn("bound_avg", col("mapping.bound_avg")) \
                       .withColumn("reaction_count", col("mapping.reaction_count")) \
                       .withColumn("pathway_name", col("mapping.pathway_name")) \
                       .drop("mapping")

    return final_df

# fpath = data_dir / datasource / f"metabolomics_test_0.parquet.gzip"
# base_df = pd.read_parquet(fpath)



metabolomics_df, label_df, _, _ = PathwayFluxMinMaxDataset.source_load_data(
    dataset_ids=test_source_list,
    datasource="aycan",
    # pathway_features=True,
)
# for column in base_df.columns:
#     if column not in metabolomics_df.columns:
#         metabolomics_df[column] = None

metabolomics_df = spark.createDataFrame(metabolomics_df)
label_df = spark.createDataFrame(label_df)

# final_df = repeat_dataset(X=metabolomics_df, y=label_df)
# final_df.to_parquet(out_dir/f"test_oneout_cancer.parquet.gzip", compression="gzip")

final_df = repeat_dataset_spark(X_spark=metabolomics_df, y_spark=label_df, mapping_dict=mapping)

fpath = out_dir/f"test_oneout_cancer.parquet.gzip"
fpath = str(fpath)
final_df.write.parquet(fpath, compression="gzip", mode="overwrite")

print(f"{metabolomics_df.count() = }", f"{label_df.count() = }", f"{final_df.count() = }")
print(f"{len(metabolomics_df.columns) = }", f"{len(label_df.columns) = }", f"{len(final_df.columns) = }")


# # metabolite_scaler_method = None
# # target_scaler_method = None
# # # metabolite_coverage = "fully"
# # metabolite_coverage = None
# # pathway_features = True
for fold in tqdm(range(k_folds)):

    # train_all_dataset = PathwayFluxMinMaxDataset(
    #     dataset_ids=[f"train_{fold}"],
    #     scaler_method=target_scaler_method,
    #     metabolite_scaler_method=metabolite_scaler_method,
    #     datasource=datasource,
    #     metabolite_coverage=metabolite_coverage,
    #     pathway_features=pathway_features,
    #     # eval_mode=False,
    #     run_init=True,
    # )
    # .load_data(dataset_ids=[source], pathway_features=pathway_features, datasource=datasource)
    fpath = data_dir / datasource / f"metabolomics_train_{fold}.parquet.gzip"
    fpath = str(fpath)
    metabolomics_df = spark.read.parquet(fpath, inferSchema=True)
    fpath = data_dir / datasource / f"label_train_{fold}.parquet.gzip"
    fpath = str(fpath)
    label_df = spark.read.parquet(fpath, inferSchema=True)

    final_df = repeat_dataset_spark(X_spark=metabolomics_df, y_spark=label_df, mapping_dict=mapping)

    fpath = out_dir/f"train_oneout_{fold}.parquet.gzip"
    fpath = str(fpath)
    final_df.write.parquet(fpath, compression="gzip", mode="overwrite")

    # final_df.to_parquet(out_dir/f"train_oneout_{fold}.parquet.gzip", compression="gzip")
    del metabolomics_df
    del label_df
    del final_df
    gc.collect()
    print(f"train done {fold = }")


    fpath = data_dir / datasource / f"metabolomics_test_{fold}.parquet.gzip"
    fpath = str(fpath)
    metabolomics_df = spark.read.parquet(fpath, inferSchema=True)
    fpath = data_dir / datasource / f"label_test_{fold}.parquet.gzip"
    fpath = str(fpath)
    label_df = spark.read.parquet(fpath, inferSchema=True)

    final_df = repeat_dataset_spark(X_spark=metabolomics_df, y_spark=label_df, mapping_dict=mapping)

    fpath = out_dir/f"test_oneout_{fold}.parquet.gzip"
    fpath = str(fpath)
    final_df.write.parquet(fpath, compression="gzip", mode="overwrite")

    print(f"{metabolomics_df.count() = }", f"{label_df.count() = }", f"{final_df.count() = }")
    print(f"{len(metabolomics_df.columns) = }", f"{len(label_df.columns) = }", f"{len(final_df.columns) = }")

    # final_df.to_parquet(out_dir/f"train_oneout_{fold}.parquet.gzip", compression="gzip")
    del metabolomics_df
    del label_df
    del final_df
    gc.collect()
    print(f"validation done {fold = }")
    # break

    # metabolomics_df, label_df, _, _ = PathwayFluxMinMaxDataset.source_load_data(
    #     dataset_ids=[f"test_{fold}"],
    #     datasource=datasource,
    #     # pathway_features=False,
    # )
    # final_df = repeat_dataset(X=metabolomics_df, y=label_df)
    # final_df.to_parquet(out_dir/f"test_oneout_{fold}.parquet.gzip", compression="gzip")
    # del metabolomics_df
    # del label_df
    # gc.collect()
    # print(f"validation done {fold = }")
