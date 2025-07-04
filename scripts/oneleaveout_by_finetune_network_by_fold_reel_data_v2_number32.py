# %%
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import random
import time

import torch
from sklearn.model_selection import KFold

from deep_metabolitics.data.properties import get_dataset_ids
from deep_metabolitics.utils.logger import create_logger
from deep_metabolitics.utils.trainer import eval_dataset, train

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# %%
from deep_metabolitics.data.metabolight_dataset import PathwayDataset
from deep_metabolitics.networks.multiout_regressor_net_v2 import MultioutRegressorNETV2

# %%
dataset_ids = get_dataset_ids()


VALIDATION_RATE = 0.2
SHUFFLE = True

epochs = 200
early_stopping_metric_name = "mse"
dropout_rate = 0.5
batch_size = 49
early_stopping_min_delta = 0.038741
early_stopping_patience = 44  # default 20
learning_rate = 0.082900
loss_method = "rmse"
n_start_layers = 3
scheduler_gamma = 0.161574
scheduler_step_size = 44
target_scaler_method = None
weight_decay = 0.000987


n_train_dataset = len(dataset_ids)


log_metrics = {
    "VALIDATION_RATE": VALIDATION_RATE,
    "SHUFFLE": SHUFFLE,
    "epochs": epochs,
    "early_stopping_patience": early_stopping_patience,
    "early_stopping_min_delta": early_stopping_min_delta,
    "early_stopping_metric_name": early_stopping_metric_name,
    "dropout_rate": dropout_rate,
    "weight_decay": weight_decay,
    "target_scaler_method": target_scaler_method,
    "learning_rate": learning_rate,
    "scheduler_step_size": scheduler_step_size,
    "scheduler_gamma": scheduler_gamma,
    "n_start_layers": n_start_layers,
    "batch_size": batch_size,
    "n_train_dataset": n_train_dataset,
    "loss_method": loss_method,
}

# %% [markdown]
# # oneleaveout_by_multiout-regressor-netv2_tuned_v2
#

# %%
experiment_name = "oneleaveout_by_finetune_network_by_fold_reel_data_v2_number32"
logger = create_logger(experiment_name, remove=False)
kf = KFold(n_splits=len(dataset_ids))
kf.get_n_splits(dataset_ids)

# %%
for i, (train_index, test_index) in enumerate(kf.split(dataset_ids)):
    experiment_fold = (
        f"{experiment_name}_fold_{i}_testindex_{dataset_ids[test_index][0]}"
    )
    logger.info(
        {
            "fold": i,
            "train_index": dataset_ids[train_index],
            "test_index": dataset_ids[test_index],
        }
    )

    start_time = time.time()
    dataset = PathwayDataset(
        dataset_ids=dataset_ids[train_index], scaler_method=target_scaler_method
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        {
            "fold": i,
            "train_dataset_load_time": elapsed_time,
            "dataset_size": len(dataset),
        }
    )

    start_time = time.time()
    test_dataset = PathwayDataset(
        dataset_ids=dataset_ids[test_index], scaler=dataset.scaler
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        {
            "fold": i,
            "test_dataset_load_time": elapsed_time,
            "test_dataset_size": len(test_dataset),
        }
    )

    n_features = dataset.n_metabolights
    out_features = dataset.n_labels

    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [1 - VALIDATION_RATE, VALIDATION_RATE]
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(batch_size), shuffle=SHUFFLE, drop_last=True
    )

    if len(dataloader) > 0 and len(test_dataset):
        model = MultioutRegressorNETV2(
            n_features=n_features,
            out_features=out_features,
            n_start_layers=n_start_layers,
            dropout_rate=dropout_rate,
            loss_method=loss_method,
        )

        model, optimizer, train_metrics, validation_metrics = train(
            epochs=epochs,
            dataloader=dataloader,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            n_start_layers=n_start_layers,
            batch_size=batch_size,
            logger=logger,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_metric_name=early_stopping_metric_name,
            fold=i,
            fname=f"{experiment_fold}.pt",
        )

        # Evaluation Validation
        eval_dataset(
            model=model,
            dataset=test_dataset,
            logger=logger,
            fold=i,
            running_for="TEST",
            # epoch=epoch,
        )

# %%
from deep_metabolitics.utils.logger import read_log_file

train_df, validation_df, test_df = read_log_file(experiment_name=experiment_name)

# %%
import os

from deep_metabolitics.config import oneleaveout_results_dir

result_dir = oneleaveout_results_dir / experiment_name
os.makedirs(result_dir, exist_ok=True)
test_df.to_csv(result_dir / f"test_results.csv")
train_df.to_csv(result_dir / f"train_results.csv")
validation_df.to_csv(result_dir / f"validation_results.csv")
