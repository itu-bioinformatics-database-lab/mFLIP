import os


from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, monotonically_increasing_id
import pyspark.sql.functions as F


from deep_metabolitics.utils.performance_metrics import PerformanceMetrics
from deep_metabolitics.config import outputs_dir

# SparkSession oluşturma
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


def get_real_target(df):
    df = df.select(["row_idx", "target_name", "target_value"]).toPandas()
    _target = df.pivot_table(
        index="row_idx",
        columns="target_name",
        values="target_value",
    )
    return _target[sorted(_target.columns)]

def get_predicted_target(df):
    df = df.select(["row_idx", "target_name", "prediction"]).toPandas()
    _target = df.pivot_table(
        index="row_idx",
        columns="target_name",
        values="prediction",
    )
    return _target[sorted(_target.columns)]

k_folds = 10


experiment_name = "pm_wm_pathwayfluxminmax_xgboost_oneout"
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

for fold in list(range(k_folds)):
    experiment_fold = f"{experiment_name}_fold_{fold}"
    print(f"{experiment_fold = }")


    fpath = os.path.join(
        outputs_dir, f"predictions_train_{experiment_fold}.parquet.gzip"
    )
    train_oo_df = spark.read.parquet(fpath)
    train_oo_df = train_oo_df.withColumn("__tmp_index", monotonically_increasing_id())
    w = Window.partitionBy("target_name").orderBy("__tmp_index")
    train_oo_df = (
        train_oo_df
        .withColumn("row_idx", row_number().over(w) - 1)  # sıfır‐tabanlı
        .drop("__tmp_index")
    )
    print(f"{train_oo_df.count() = }")
    train_oo_df.show()
    train_real_target = get_real_target(df=train_oo_df)
    train_predicted_target = get_predicted_target(df=train_oo_df)


    fpath = os.path.join(
        outputs_dir, f"predictions_validation_{experiment_fold}.parquet.gzip"
    )
    validation_oo_df = spark.read.parquet(fpath)
    validation_oo_df = validation_oo_df.withColumn("__tmp_index", monotonically_increasing_id())
    w = Window.partitionBy("target_name").orderBy("__tmp_index")
    validation_oo_df = (
        validation_oo_df
        .withColumn("row_idx", row_number().over(w) - 1)  # sıfır‐tabanlı
        .drop("__tmp_index")
    )
    validation_real_target = get_real_target(df=validation_oo_df)
    validation_predicted_target = get_predicted_target(df=validation_oo_df)


    fpath = os.path.join(
        outputs_dir, f"predictions_test_{experiment_fold}.parquet.gzip"
    )
    test_oo_df = spark.read.parquet(fpath)
    test_oo_df = test_oo_df.withColumn("__tmp_index", monotonically_increasing_id())
    w = Window.partitionBy("target_name").orderBy("__tmp_index")
    test_oo_df = (
        test_oo_df
        .withColumn("row_idx", row_number().over(w) - 1)  # sıfır‐tabanlı
        .drop("__tmp_index")
    )
    test_real_target = get_real_target(df=test_oo_df)
    test_predicted_target = get_predicted_target(df=test_oo_df)


    print(f"{train_oo_df.count() = }")
    print(f"{validation_oo_df.count() = }")
    print(f"{test_oo_df.count() = }")

    performance_metrics = PerformanceMetrics(
        target_names=list(train_real_target.columns),
        experience_name=experiment_fold,
        train_time=0,
        test_time=0,
        validation_time=0,
        scaler=None,
    )
    performance_metrics.train_metric(y_true=train_real_target.values, y_pred=train_predicted_target.values)
    performance_metrics.validation_metric(
        y_true=validation_real_target.values, y_pred=validation_predicted_target.values
    )
    performance_metrics.test_metric(y_true=test_real_target.values, y_pred=test_predicted_target.values)
    performance_metrics.complete()  # TODO foldlari tek dosyada tutsak guzel olur
