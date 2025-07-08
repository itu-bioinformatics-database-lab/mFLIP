import warnings
import os

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import random
import time
import torch
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel

from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from pytorch_tabular.utils import make_mixed_dataset, print_metrics


from deep_metabolitics.config import all_generated_datasets_dir, aycan_full_data_dir

# from deep_metabolitics.data.properties import get_dataset_ids

from deep_metabolitics.data.metabolight_dataset import (
    ReactionMinMaxDataset,
)
from deep_metabolitics.data.properties import get_all_ds_ids

from pytorch_tabular import TabularModel
from pytorch_tabular.models import DANetConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from lion_pytorch import Lion
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.set_float32_matmul_precision('high')  # Alternatif: 'medium'
# 'high' → En iyi performans, daha düşük hassasiyet.
# 'medium' → Performans ve hassasiyet arasında dengeli seçim.
# 'highest' (Varsayılan) → En yüksek hassasiyet, ancak daha yavaş.
# 🚀 Önerim: "high" modunu kullanarak hız kazanabilirsin.


experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

metabolite_coverage = "aycan_union"

metabolite_scaler_method = None
target_scaler_method = None

aycan_source_list = [
    "metabData_breast",
    "metabData_ccRCC3",
    "metabData_ccRCC4",
    "metabData_coad",
    "metabData_pdac",
    "metabData_prostat",
]

generated_ds_ids = get_all_ds_ids(folder_path=all_generated_datasets_dir)
uniform_dataset = ReactionMinMaxDataset(
    dataset_ids=generated_ds_ids,
    scaler_method=target_scaler_method,
    metabolite_scaler_method=metabolite_scaler_method,
    datasource="all_generated_datasets",
    metabolite_coverage=metabolite_coverage,
    pathway_features=False,
)



print(f"{uniform_dataset.metabolomics_df.isna().sum().sum()}")
print(f"{uniform_dataset.metabolomics_df.shape}")
print(f"{uniform_dataset.label_df.isna().sum().sum()}")
print(f"{uniform_dataset.label_df.shape}")