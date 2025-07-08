import warnings

warnings.filterwarnings("ignore")

import os
import sys
import random
import pandas as pd

from deep_metabolitics.data.properties import get_workbench_metabolights_dataset_ids
from deep_metabolitics.config import work_workbench_metabolights_multiplied_by_factors_dir
from deep_metabolitics.data.metabolight_dataset import (
    PathwayDataset,
    PathwayFluxMinMaxDataset,
    PathwayMinMaxDataset,
    ReactionMinMaxDataset,
)

seed = 10
random.seed(seed)
metabolite_scaler_method=None
target_scaler_method=None
metabolite_coverage="fully"

data_source_list = get_workbench_metabolights_dataset_ids()

train_dataset = ReactionMinMaxDataset(
    dataset_ids=data_source_list,
    scaler_method=target_scaler_method,
    metabolite_scaler_method=metabolite_scaler_method,
    datasource="workbench_metabolights",
    metabolite_coverage=metabolite_coverage,
    pathway_features=False,
)


size_MB = sys.getsizeof(train_dataset) / (1024 * 1024)
print(f"Objenin boyutu: {size_MB:.6f} MB")