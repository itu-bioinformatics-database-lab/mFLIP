import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, monotonically_increasing_id
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, count, isnan, lit

from deep_metabolitics.data.properties import get_recon_metabolites

all_metabolities = get_recon_metabolites()


import torch
from torch.utils.data import Dataset


spark = SparkSession.builder \
    .appName("BigDataProcessing") \
    .master("local[18]") \
    .config("spark.driver.memory", "32g") \
    .config("spark.sql.shuffle.partitions", "600") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "100000") \
    .config("spark.local.dir", "/tmp/bacan_disk_spark") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/bacan_spark-checkpoint")
spark.conf.set('spark.sql.caseSensitive', True)


class OneoutDataset(Dataset):
    def __init__(self, dataframe_path, scaler, one_hot_encoder=None, features=None, filter_null_columns_by=0.9, filter_null_rows_by=0.75):
        print(f"{dataframe_path = }")
        self.all_metabolities = set(get_recon_metabolites())
        if isinstance(dataframe_path, DataFrame):
            self.dataframe = dataframe_path
        else:
            self.dataframe = spark.read.parquet(str(dataframe_path), inferSchema=True)
            # self.dataframe = spark.read.parquet(str(dataframe_path), inferSchema=True)
            print(f"{dataframe_path = }")
        total_rows = self.dataframe.count()
        print(f"before fillna {total_rows = }")
        if features is None:
            self.working_metabolites = list(self.all_metabolities.intersection(self.dataframe.columns))
            self.features = self.working_metabolites + ["bound_avg", "reaction_count"]
            self.features = sorted(self.features)

            null_ratios = (
                self.dataframe.select([
                    (count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in self.features
                ])
            ).first().asDict()
            self.features = sorted([col_name for col_name, null_ratio in null_ratios.items() if null_ratio < filter_null_columns_by])
        else:
            self.features = features

        total_columns = len(self.features)
        # Satır bazında null oranı hesapla
        row_null_ratio_expr = (
            sum(when(col(c).isNull(), 1).otherwise(0) for c in self.features) / lit(total_columns)
        )
        
        self.dataframe = self.dataframe.withColumn("null_ratio", row_null_ratio_expr)
        self.dataframe = self.dataframe.filter(col("null_ratio") <= filter_null_rows_by).drop("null_ratio")

        # self.dataframe.select(['target_name','target_value']).show()
        self.dataframe = self.dataframe.fillna(0.0)
        # self.dataframe = self.dataframe.limit(10_000)
        # self.dataframe["row_idx"] = self.dataframe.groupby("target_name").cumcount()
        print(f"after fillna {self.dataframe.count()}")


        # 1) Orijinal sıralamayı korumak için geçici bir index ekleyin
        self.dataframe = self.dataframe.withColumn("__tmp_index", monotonically_increasing_id())

        # 2) target_name'e göre partition oluşturup, __tmp_index ile sıralama yapan window
        w = Window.partitionBy("target_name").orderBy("__tmp_index")

        # 3) row_number() ile 1‐based sayacı ekleyin, istersen 0‐based yapmak için -1 çıkarabilirsiniz
        self.dataframe = (
            self.dataframe
            .withColumn("row_idx", row_number().over(w) - 1)  # sıfır‐tabanlı
            .drop("__tmp_index")
        )
        print(f"after row_idx {self.dataframe.count()}")

        # 1) Sadece bir kere (örneğin Dataset.__init__ içinde) yapın:
        w = Window.orderBy(monotonically_increasing_id())
        self.dataframe = (
            self.dataframe
                .withColumn("idx", row_number().over(w) - 1)  # 0‐based index
                .cache()
        )
        print(f"after idx {self.dataframe.count()}")

        self.dataframe = self.dataframe.select(self.features + ["row_idx", "idx", "pathway_name", "target_value", "target_name"]).toPandas()
        spark.stop()

        self.scaler = self.fit_scaler(scaler)

        if one_hot_encoder is None:
            one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            one_hot_encoder.fit(self.dataframe["pathway_name"].values)

        self.one_hot_encoder = one_hot_encoder

        # self.X = self.dataframe.drop(columns=["target", "value"]).values.astype("float32")
        # self.y = self.dataframe["target_value"].values.astype("float32")

    def fit_scaler(self, scaler):
        if not  isinstance(scaler, str):
            return scaler

        if scaler == "minmax":
            scaler = MinMaxScaler()
            # scaler = MinMaxScaler(feature_range=(-10, 10))
        elif scaler == "std":
            scaler = StandardScaler()
        self.scaler.fit(self.dataframe[self.features])
        return scaler



    def __len__(self):
        return len(self.dataframe)
    
    @property
    def n_features(self):
        return len(self.features) + 98

    @property
    def n_labels(self):
        return 1

    def __getitem__(self, idx):

        features = self.dataframe.iloc[idx]

        # features = self.dataframe.iloc[idx]
        target = features["target_value"].values
        # print(f"{target = }")

        target = torch.tensor(target, dtype=torch.float32)
        # print(f"{target = }")
        

        for f in self.features:
            if f not in features.columns:
                features[f] = 0
        features = features.fillna(0)
        # print("before onehotencoder ####")
        # print(f"{features['pathway_name'] = }")
        ohe_features = self.one_hot_encoder.transform(features[["pathway_name"]])
        # print(f"{ohe_features = }")

        # print(f"{self.scaler.mean_ = }")
        # print(f"{self.scaler.var_ = }")
        features = self.scaler.transform(features[self.features])
        # print(f"{features = }")

        features = torch.tensor(features, dtype=torch.float32)
        # print(f"{features = }")
        ohe_features = torch.tensor(ohe_features, dtype=torch.float32)
        # print(f"{ohe_features = }")

        # print(f"{features.shape = }")
        # print(f"{ohe_features.shape = }")
        features = torch.cat([features, ohe_features], dim=1)

        # print(f"{features = }", "before return")

        features = features.squeeze(0)


        return features, target

    def set_predicted(self, predicted):
        self.dataframe["predicted"] = predicted

    def get_real_target(self):
        df = self.dataframe[["row_idx", "target_name", "target_value"]]
        _target = df.pivot_table(
            index="row_idx",
            columns="target_name",
            values="target_value",
        )
        return _target[sorted(_target.columns)]

    def get_predicted_target(self):
        df = self.dataframe[["row_idx", "target_name", "predicted"]]
        _target = df.pivot_table(
            index="row_idx",
            columns="target_name",
            values="predicted",
        )
        return _target[sorted(_target.columns)]