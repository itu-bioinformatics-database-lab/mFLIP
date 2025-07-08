import warnings

warnings.filterwarnings("ignore")

import os
import random
import pandas as pd

from deep_metabolitics.data.properties import get_workbench_metabolights_dataset_ids
from deep_metabolitics.config import work_workbench_metabolights_multiplied_by_factors_dir


seed = 10
random.seed(seed)


data_source_list = get_workbench_metabolights_dataset_ids()


factors_df_list = []
metabolomics_df_list = []
label_df_list = []
dataset_ids_df_list = []

for dataset_id in data_source_list:
    factors_df = pd.read_csv(
        work_workbench_metabolights_multiplied_by_factors_dir
        / f"{dataset_id}.csv"
    )[["Factors"]]

    metabolomics_df = pd.read_csv(
        work_workbench_metabolights_multiplied_by_factors_dir
        / f"foldchange_{dataset_id}.csv",
        index_col=0,
    )

    label_df = pd.read_csv(
        work_workbench_metabolights_multiplied_by_factors_dir
        / f"fluxminmax_{dataset_id}.csv",
        index_col=0,
    )

    values = [dataset_id] * len(metabolomics_df)
    dataset_ids_df = pd.DataFrame(
        values, index=metabolomics_df.index, columns=["dataset_id"]
    )

    factors_df_list.append(factors_df)
    dataset_ids_df_list.append(dataset_ids_df)
    metabolomics_df_list.append(metabolomics_df)
    label_df_list.append(label_df)

factors_df = pd.concat(factors_df_list).reset_index(drop=True)
metabolomics_df = pd.concat(metabolomics_df_list).reset_index(drop=True)
label_df = pd.concat(label_df_list).reset_index(drop=True)
dataset_ids_df = pd.concat(dataset_ids_df_list).reset_index(drop=True)



print("factors_df", factors_df.info(memory_usage="deep"))
print("metabolomics_df", metabolomics_df.info(memory_usage="deep"))
print("label_df", label_df.info(memory_usage="deep"))
print("dataset_ids_df", dataset_ids_df.info(memory_usage="deep"))
