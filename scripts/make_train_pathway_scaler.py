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
from pytorch_tabular.models import CategoryEmbeddingModelConfig
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
    PathwayMinMaxDataset,
)
from deep_metabolitics.data.properties import get_all_ds_ids

from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from lion_pytorch import Lion
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

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

# generated_ds_ids = get_all_ds_ids(folder_path=all_generated_datasets_dir)
generated_ds_ids = [
        fname.replace("pathway_", "").replace(".csv", "")
        for fname in os.listdir(all_generated_datasets_dir)
        if "pathway_" in fname
    ]
uniform_dataset = PathwayMinMaxDataset(
    dataset_ids=generated_ds_ids,
    scaler_method=target_scaler_method,
    metabolite_scaler_method=metabolite_scaler_method,
    datasource="all_generated_datasets",
    metabolite_coverage=metabolite_coverage,
    pathway_features=False,
)


train_pathway_df_list = []

for ds_id in generated_ds_ids:
    train_pathway_df = pd.read_csv(
        all_generated_datasets_dir / f"pathway_{ds_id}.csv",
        index_col=0,
    )
    train_pathway_df_list.append(train_pathway_df)

train_pathway_df = pd.concat(train_pathway_df_list)
train_pathway_scaler = StandardScaler().fit(train_pathway_df)

import joblib

joblib.dump(train_pathway_scaler, "train_pathway_scaler.pkl")

# train_pathway_scaler'ı geri yükleme
loaded_train_pathway_scaler = joblib.load("train_pathway_scaler.pkl")

# Kullanım
# X_scaled = loaded_train_pathway_scaler.transform(X_test)
print("TAMAMLANDI")