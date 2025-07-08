import warnings

from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset

warnings.filterwarnings("ignore")

import os
import gc
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from pyspark.sql import SparkSession


from deep_metabolitics.data.properties import get_workbench_metabolights_dataset_ids
from deep_metabolitics.config import data_dir
seed = 10
random.seed(seed)


experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

out_dir = data_dir / "reactionminmax_10_folds_pyspark"

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
pathway_features = True
datasource = "workbench_metabolights"

source_list = get_workbench_metabolights_dataset_ids()
map = {}
fold_temp_datasets = dict()

print(f"{len(source_list) = }")

spark = SparkSession.builder.appName("PandasToParquet").config("spark.driver.memory", "128g").getOrCreate()

for source in tqdm(source_list):
    if source not in filter_ds:

        metabolomics_df, label_df, dataset_ids_df, factors_df = (
            ReactionMinMaxDataset.load_data_workbench_metabolights_csv(dataset_ids=[source])
        )
        # print(source, f"{metabolomics_df.shape = }", f"{factors_df.shape = }")

        # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        try:
            fold_indices = list(kf.split(metabolomics_df, factors_df))
        except Exception as e:
            train_idx = list(np.arange(len(metabolomics_df)))
            test_idx = []
            fold_indices = [[train_idx, test_idx]] * k_folds
            print(source, e, "All of used as train dataset")
        map[source] = {}
        map[source]["k_fold"] = kf
        map[source]["fold_indices"] = fold_indices
print("Fold indices are created !!!")
for fold in tqdm(range(k_folds)):
    fold_temp_datasets = dict()
    for source in map.keys():
        metabolomics_df, label_df, dataset_ids_df, factors_df = (
            ReactionMinMaxDataset.load_data_workbench_metabolights_csv(dataset_ids=[source])
        )
        fold_index = map[source]["fold_indices"][fold]
#         for fold, fold_index in enumerate(fold_indices):
#             map[source][fold] = {}
        train_indices = fold_index[0]
        test_indices = fold_index[1]
        if fold not in fold_temp_datasets:
            fold_temp_datasets[fold] = {}
            fold_temp_datasets[fold]["train_metabolomics_df"] = []
            fold_temp_datasets[fold]["train_label_df"] = []
            fold_temp_datasets[fold]["train_factors_df"] = []
            fold_temp_datasets[fold]["train_dataset_ids_df"] = []

            fold_temp_datasets[fold]["test_metabolomics_df"] = []
            fold_temp_datasets[fold]["test_label_df"] = []
            fold_temp_datasets[fold]["test_factors_df"] = []
            fold_temp_datasets[fold]["test_dataset_ids_df"] = []

        fold_temp_datasets[fold]["train_metabolomics_df"].append(
            metabolomics_df.loc[train_indices]
        )
        fold_temp_datasets[fold]["train_label_df"].append(
            label_df.loc[train_indices]
        )
        fold_temp_datasets[fold]["train_factors_df"].append(
            factors_df.loc[train_indices]
        )
        fold_temp_datasets[fold]["train_dataset_ids_df"].append(
            dataset_ids_df.loc[train_indices]
        )

        fold_temp_datasets[fold]["test_metabolomics_df"].append(
            metabolomics_df.loc[test_indices]
        )
        fold_temp_datasets[fold]["test_label_df"].append(
            label_df.loc[test_indices]
        )
        fold_temp_datasets[fold]["test_factors_df"].append(
            factors_df.loc[test_indices]
        )
        fold_temp_datasets[fold]["test_dataset_ids_df"].append(
            dataset_ids_df.loc[test_indices]
        )



    metabolomics_df=pd.concat(
            fold_temp_datasets[fold]["train_metabolomics_df"]
        )
    spark_df = spark.createDataFrame(metabolomics_df)
    save_path = str(out_dir/f"metabolomics_train_{fold}.parquet")
    spark_df.write.parquet(save_path)

    label_df=pd.concat(fold_temp_datasets[fold]["train_label_df"])
    spark_df = spark.createDataFrame(label_df)
    save_path = str(out_dir/f"label_train_{fold}.parquet")
    spark_df.write.parquet(save_path)

    factors_df=pd.concat(fold_temp_datasets[fold]["train_factors_df"])
    spark_df = spark.createDataFrame(factors_df)
    save_path = str(out_dir/f"factors_train_{fold}.parquet")
    spark_df.write.parquet(save_path)

    dataset_ids_df=pd.concat(fold_temp_datasets[fold]["train_dataset_ids_df"])
    spark_df = spark.createDataFrame(dataset_ids_df)
    save_path = str(out_dir/f"dataset_ids_train_{fold}.parquet")
    spark_df.write.parquet(save_path)

    metabolomics_df=pd.concat(
        fold_temp_datasets[fold]["test_metabolomics_df"]
    )
    spark_df = spark.createDataFrame(metabolomics_df)
    save_path = str(out_dir/f"metabolomics_test_{fold}.parquet")
    spark_df.write.parquet(save_path)

    label_df=pd.concat(fold_temp_datasets[fold]["test_label_df"])
    spark_df = spark.createDataFrame(label_df)
    save_path = str(out_dir/f"label_test_{fold}.parquet")
    spark_df.write.parquet(save_path)

    factors_df=pd.concat(fold_temp_datasets[fold]["test_factors_df"])
    spark_df = spark.createDataFrame(factors_df)
    save_path = str(out_dir/f"factors_test_{fold}.parquet")
    spark_df.write.parquet(save_path)

    dataset_ids_df=pd.concat(fold_temp_datasets[fold]["test_dataset_ids_df"])
    spark_df = spark.createDataFrame(dataset_ids_df)
    save_path = str(out_dir/f"dataset_ids_test_{fold}.parquet")
    spark_df.write.parquet(save_path)

    del fold_temp_datasets
    gc.collect()
