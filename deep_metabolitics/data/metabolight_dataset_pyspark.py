import os


import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.types import FloatType
from pyspark import StorageLevel

from deep_metabolitics.config import data_dir


class MultioutDataset(Dataset):
    def __init__(self,spark, metabolite_fpath, label_fpath, batch_size=32, **kwargs):
        # self.spark = SparkSession.builder.appName("ParquetToPyTorch").config("spark.driver.memory", "128g").config("spark.executor.memory", "64g").getOrCreate()
        self.spark = spark
        self.spark.conf.set('spark.sql.caseSensitive', True)
        self.kwargs = kwargs

        self.df = self.load(spark=self.spark, fpath=metabolite_fpath)
        print("Metabolites are loaded.")
        self.label_df = self.load(spark=self.spark, fpath=label_fpath)
        print("Labels are loaded.")
        self.preprocess()
        self.batch_size = batch_size
        self.num_rows = self.df.count()  # Toplam satır sayısı
        self.n_metabolights = len(self.metabolite_names)
        self.n_labels = len(self.label_names)

    @staticmethod
    def load(spark, fpath):
        

        # Parquet dosyasını okuma
        if not isinstance(fpath, pd.DataFrame):
            fpath = str(fpath)
            print(f"{fpath = }")
            # df = pd.read_parquet(fpath)
            # Sütun isimlerinin tekrar edip etmediğini kontrol et
            # duplicates = df.columns[df.columns.duplicated()]

            # if len(duplicates) > 0:
            #     print("Tekrarlanan sütunlar:", list(duplicates))
            # else:
            #     print("Tüm sütun isimleri benzersiz.")
            # df = spark.read.option("columnNameOfCorruptRecord", "_corrupt_record").parquet(fpath)
            df = spark.read.parquet(fpath, inferSchema=True)
            # print(len(pd.read_parquet(fpath).columns))

            # print(df.columns)
            # for c in df.columns:
            #     print(c)
            # print(df.columns)
        else:
            df = spark.createDataFrame(fpath)

        df.printSchema()
        return df
    
    def add_pathway_features(self):
        from deep_metabolitics.utils.utils import load_pathway_metabolites_map


        self.pathway_features = self.kwargs.get("pathway_features", False)
        if self.pathway_features:
            self.pathway_metabolites_columns = []
            pathway_metabolites_map = load_pathway_metabolites_map(is_unique=True)
            for pathway_name, metabolities in pathway_metabolites_map.items():
                intersect_metabolities = set(self.df.columns).intersection(
                    metabolities
                )
                sum_expr = sum([F.when(F.isnull(F.col(col)), 0).otherwise(F.col(col)) for col in intersect_metabolities])
                count_expr = sum([F.when(F.isnull(F.col(col)), 0).otherwise(1) for col in intersect_metabolities])
                avg_expr = sum_expr / count_expr

                new_col = f"{pathway_name}_mean"

                self.df = self.df.withColumn(new_col, avg_expr)

                self.pathway_metabolites_columns.append(new_col)
    
    def _preprocess_metabolites(self, metabolities):
        self.metabolite_model = self.kwargs.get("metabolite_model")

        for m in metabolities:
            if m not in self.df.columns:
                self.df = self.df.withColumn(m, F.lit(0.0).cast(FloatType()))

        # Imputation with 0
        self.df = self.df.na.fill(0.0)

        if self.metabolite_model is None:
            # 1. Adım: Feature'ları tek bir vektör halinde birleştirmek için VectorAssembler kullanıyoruz
            assembler = VectorAssembler(inputCols=metabolities, outputCol="features")
            # 2. Adım: Özellikleri standartlaştırmak için StandardScaler kullanıyoruz
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

            # 3. Adım: Bu adımları bir Pipeline içinde birleştiriyoruz
            pipeline = Pipeline(stages=[assembler, scaler])

            # Eğitim verisi üzerinde fit işlemi yapıyoruz
            self.metabolite_model = pipeline.fit(self.df)
        
        self.df = self.metabolite_model.transform(self.df)
        self.df.persist(StorageLevel.DISK_ONLY)  # Belleğe sığmayacaksa sadece diskte sakla
        self.df = self.df.checkpoint()  # Büyük ara çıktıları diske kaydet
        self.metabolite_names = metabolities
        self.feature_columns = "scaled_features"


    def _preprocess_labels(self, labels):
        self.label_model = self.kwargs.get("label_model")
        # Imputation with 0
        self.label_df = self.label_df.na.fill(0.0)

        if self.label_model is None:
            # # 1. Adım: Feature'ları tek bir vektör halinde birleştirmek için VectorAssembler kullanıyoruz
            # assembler = VectorAssembler(inputCols=labels, outputCol="labels")
            assemblers = [VectorAssembler(inputCols=[label], outputCol=f"vector_{label}") for label in labels]
            scalers = [StandardScaler(inputCol=f"vector_{label}", outputCol=f"scaled_{label}") for label in labels]
            # # 2. Adım: Özellikleri standartlaştırmak için StandardScaler kullanıyoruz
            # scaler = StandardScaler(inputCol="labels", outputCol="scaled_labels")

            # 3. Adım: Bu adımları bir Pipeline içinde birleştiriyoruz
            # pipeline = Pipeline(stages=[assembler, scaler])
            pipeline = Pipeline(stages=assemblers + scalers)

            # Eğitim verisi üzerinde fit işlemi yapıyoruz
            self.label_model = pipeline.fit(self.label_df)
        
        self.label_df = self.label_model.transform(self.label_df)
        self.label_df.persist(StorageLevel.DISK_ONLY)  # Belleğe sığmayacaksa sadece diskte sakla
        self.label_df = self.label_df.checkpoint()  # Büyük ara çıktıları diske kaydet
        self.label_names = labels



    def preprocess(self):
        from deep_metabolitics.data.properties import get_aycan_and_db_union_metabolites, get_aycan_union_metabolites, get_recon_metabolites

        from deep_metabolitics.utils.utils import load_pathway_metabolites_map

        # self.add_pathway_features()
        self.pathway_features = None

        self.metabolite_coverage = self.kwargs.get("metabolite_coverage", None)

        

        if self.metabolite_coverage == "fully":
            metabolites = get_recon_metabolites()
        elif self.metabolite_coverage == "aycan_union":
            metabolites = get_aycan_union_metabolites()
        elif self.metabolite_coverage == "aycan_union_plus_pathways":
            metabolites = get_aycan_union_metabolites()
            # metabolites += list(self.label_df.columns)
        elif self.metabolite_coverage == "db_aycan_union":
            metabolites = get_aycan_and_db_union_metabolites()
        elif self.metabolite_coverage == "mm_union":
            from deep_metabolitics.data.properties import get_workbench_metabolights_union_metabolites
            metabolites = get_workbench_metabolights_union_metabolites()
        else:
            metabolites = self.df.columns

        if self.pathway_features:
            metabolites += self.pathway_metabolites_columns
        
        metabolites = sorted(metabolites)
        label_list = sorted(self.label_df.columns)

        self._preprocess_metabolites(metabolities=metabolites)
        self._preprocess_labels(labels=label_list)



    def __len__(self):
        return self.df.count()
        # return int(np.ceil(self.num_rows / self.batch_size))

    def __getitem__(self, idx):
        # Belirli bir batch’i alıyoruz
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_rows)

        # PySpark DataFrame'i pandas DataFrame'e dönüştür
        batch_df = self.df.limit(end).subtract(self.df.limit(start)).toPandas()

        # Pandas DataFrame'den numpy array oluştur
        features = batch_df["scaled_features"].values
        labels = batch_df["label_column"].values  # Etiket sütununuzu buraya koyun

        # PyTorch Tensor'larına dönüştür
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return features_tensor, labels_tensor
