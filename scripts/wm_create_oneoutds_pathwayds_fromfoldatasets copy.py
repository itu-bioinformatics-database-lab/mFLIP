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


from deep_metabolitics.data.properties import get_aycan_dataset_ids
from deep_metabolitics.data.properties import get_workbench_metabolights_dataset_ids
from deep_metabolitics.config import data_dir
from deep_metabolitics.utils.utils import (
    load_cobra_network,
)
seed = 10
random.seed(seed)


experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

out_dir = data_dir / "reactionminmax_10_folds"

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


def get_reactionbased_mappings():
    mapping = {}
    recon3 = load_cobra_network()

    for reaction in recon3.reactions:
        reaction_name = reaction.id
        lower_bound, upper_bound = reaction.bounds
        pathway_name = reaction.subsystem
        mapping[reaction_name] = [lower_bound, upper_bound, pathway_name]
    return mapping


def repeat_dataset(X, y):
    # Feature'ları target sayısı kadar çoğalt
    X_repeat = pd.concat([X] * y.shape[1], ignore_index=True)

    # Target'ları uzun formata getir
    y_melt = y.melt(var_name="target_name", value_name="target_value")

    # Birleştir
    final_df = pd.concat([X_repeat, y_melt], axis=1)
    mapping = get_reactionbased_mappings()

    final_df[['lower_bound', 'upper_bound', 'pathway_name']] = final_df['target_name'].map(mapping).apply(pd.Series)
    return final_df

metabolomics_df, label_df, dataset_ids_df, factors_df = ReactionMinMaxDataset.load_data(
    dataset_ids=test_source_list,
    datasource="aycan",
    pathway_features=True,
)
final_df = repeat_dataset(X=metabolomics_df, y=label_df)
final_df.to_parquet(out_dir/f"test_oneout_cancer.parquet.gzip", compression="gzip")



for fold in tqdm(range(k_folds)):
    # .load_data(dataset_ids=[source], pathway_features=pathway_features, datasource=datasource)
    metabolomics_df, label_df, dataset_ids_df, factors_df = ReactionMinMaxDataset.load_data(
        dataset_ids=[f"train_{fold}"],
        datasource="reactionminmax_10_folds",
        pathway_features=False,
    )
    final_df = repeat_dataset(X=metabolomics_df, y=label_df)
    final_df.to_parquet(out_dir/f"train_oneout_{fold}.parquet.gzip", compression="gzip")

    metabolomics_df, label_df, dataset_ids_df, factors_df = ReactionMinMaxDataset.load_data(
        dataset_ids=[f"test_{fold}"],
        datasource="reactionminmax_10_folds",
        pathway_features=False,
    )
    final_df = repeat_dataset(X=metabolomics_df, y=label_df)
    final_df.to_parquet(out_dir/f"test_oneout_{fold}.parquet.gzip", compression="gzip")
    gc.collect()
