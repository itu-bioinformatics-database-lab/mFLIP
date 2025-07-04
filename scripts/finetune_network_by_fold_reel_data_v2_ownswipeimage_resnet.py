# %%
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import random

import torch

from deep_metabolitics.data.metabolight_dataset import PathwayDataset
from deep_metabolitics.data.properties import get_dataset_ids_from_csv
from deep_metabolitics.utils.logger import create_logger

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# %%
VALIDATION_RATE = 0.2
SHUFFLE = True


# %%
experiment_name = "finetune_network_by_fold_reel_data_v2_ownswipeimage_resnet"
logger = create_logger(experiment_name, remove=True)

# %%
dataset_ids = get_dataset_ids_from_csv()
# print(dataset_ids)

# %%
from sklearn.model_selection import KFold

# %%
from deep_metabolitics.networks.multiout_regressor_resnet_fow_own_swipe_image import (
    MultioutRegressorResNet,
)

kf = KFold(n_splits=len(dataset_ids))
kf.get_n_splits(dataset_ids)
for i, (train_index, test_index) in enumerate(kf.split(dataset_ids)):
    break

# %%


from deep_metabolitics.utils.trainer import train


def train_optimize(
    epochs,
    learning_rate,
    weight_decay,
    dropout_rate,
    n_start_layers,
    batch_size,
    target_scaler_method,
    early_stopping_patience,
    early_stopping_min_delta,
    early_stopping_metric_name,
    scheduler_step_size,
    scheduler_gamma,
    loss_method,
    resnet_pretrained,
    resnet_version,
):
    try:
        seed = 10
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        tune_name = f"epochs_{epochs}_learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}_target_scaler_method_{target_scaler_method}_early_stopping_patience_{early_stopping_patience}_early_stopping_min_delta_{early_stopping_min_delta}_early_stopping_metric_name_{early_stopping_metric_name}_scheduler_step_size_{scheduler_step_size}_scheduler_gamma_{scheduler_gamma}_loss_method_{loss_method}"
        logger.info({"tune_info": tune_name})

        dataset = PathwayDataset(
            dataset_ids=dataset_ids[train_index],
            scaler_method=target_scaler_method,
            is_make_image=True,
            image_converter_type="ownswipe",
            datasource="csv",
        )

        # n_features = dataset.n_metabolights
        out_features = dataset.n_labels
        n_dataset = len(dataset)
        n_validation = int(n_dataset * VALIDATION_RATE)
        n_train = n_dataset - n_validation

        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_validation]
        )

        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=int(batch_size), shuffle=SHUFFLE
        )

        model = MultioutRegressorResNet(
            out_features=out_features,
            resnet_pretrained=resnet_pretrained,
            resnet_version=resnet_version,
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
            fold=tune_name,
        )
        train_r2 = [metrics["r2"] for metrics in train_metrics]
        train_loss = [metrics[loss_method] for metrics in train_metrics]
        validation_r2 = [metrics["r2"] for metrics in validation_metrics]
        validation_loss = [metrics["r2"] for metrics in validation_metrics]

        last_result = validation_r2[-1]
        # TODO buna bir bakacagim. belki loss uzerinden gidebilirim.
        logger.info(
            {
                "tune_name": tune_name,
                "last_result": last_result,
                "train_r2": train_r2,
                "train_loss": train_loss,
                "validation_r2": validation_r2,
                "validation_loss": validation_loss,
            }
        )
    except Exception as e:
        logger.error({"error": str(e)})
        return None, None
    return train_metrics, validation_metrics


# %%
import optuna


def objective(trial):

    epochs = 200
    # early_stopping_patience = 20
    # early_stopping_min_delta = 0.001
    early_stopping_metric_name = "mse"
    # scheduler_step_size = 10
    # scheduler_step_size = 10
    # scheduler_gamma = 0.1

    dropout_rate = None
    batch_size = 50
    early_stopping_min_delta = 0.039436
    early_stopping_patience = 20

    # learning_rate = 0.082900
    n_start_layers = None
    # scheduler_gamma = 0.161574
    # scheduler_step_size = 36
    # loss_method = "rmse"
    # weight_decay = 0.0005
    # target_scaler_method = "minmax"

    learning_rate = trial.suggest_uniform("learning_rate", 0.001, 0.5)
    # early_stopping_patience = trial.suggest_int("early_stopping_patience", 5, 50)
    # early_stopping_min_delta = trial.suggest_uniform(
    #     "early_stopping_min_delta", 1e-5, 1e-1
    # )

    scheduler_step_size = trial.suggest_int("scheduler_step_size", 10, 50)
    scheduler_gamma = trial.suggest_uniform("scheduler_gamma", 0.01, 0.5)
    weight_decay = trial.suggest_uniform("weight_decay", 0, 0.01)
    target_scaler_method = trial.suggest_categorical(
        "target_scaler_method", ["minmax", "std", None]
    )  # 2
    # n_start_layers = trial.suggest_int("n_start_layers", 1, 16)
    # batch_size = trial.suggest_int("batch_size", 8, 64)  # 2
    loss_method = trial.suggest_categorical("loss_method", ["mae", "rmse"])

    # learning_rate = trial.suggest_categorical("learning_rate", [0.1, 0.01])  # 5
    # scheduler_step_size = trial.suggest_categorical(
    #     "scheduler_step_size", [5, 10, 20]
    # )  # 3
    # scheduler_gamma = trial.suggest_categorical("scheduler_gamma", [0.1, 0.5])  # 2
    # learning_rate = 0.1
    # weight_decay = trial.suggest_categorical("weight_decay", [0.0002, 0.0005])  # 2
    # weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    # weight_decay = 0.0004
    # dropout_rate = trial.suggest_categorical("dropout_rate", [0.3, 0.5])  # 2
    # dropout_rate = 0.4
    # n_start_layers = trial.suggest_categorical("n_start_layers", [2, 8, 16])  # 4
    # batch_size = trial.suggest_int("batch_size", 8, 64, 8) # 8
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])  # 2
    # loss_method = trial.suggest_categorical("loss_method", ["mse", "mae", "rmse"])  # 2
    # target_scaler_method = trial.suggest_categorical(
    #     "target_scaler_method", ["minmax", "std", None]
    # )  # 2
    resnet_pretrained = trial.suggest_categorical("resnet_pretrained", [True, False])
    resnet_version = trial.suggest_categorical("resnet_version", [18, 50])

    # Modeli oluştur ve eğit
    train_metrics, validation_metrics = train_optimize(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        n_start_layers=n_start_layers,
        batch_size=batch_size,
        target_scaler_method=target_scaler_method,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_metric_name=early_stopping_metric_name,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        loss_method=loss_method,
        resnet_pretrained=resnet_pretrained,
        resnet_version=resnet_version,
    )
    if train_metrics is not None:
        validation_r2 = validation_metrics[-1]["r2"]
        train_r2 = train_metrics[-1]["r2"]
        validation_r2 = validation_metrics[-1]["r2"]
        train_r2 = train_metrics[-1]["r2"]
        # Amaç fonksiyonu: doğrulama doğruluğunu maksimize ederken farkı minimize et
        alpha = 0.4  # Doğruluk farkını ne kadar önemseyeceğinizi belirleyin
        score = validation_r2 - alpha * (train_r2 - validation_r2)

        # Optuna denemesinde doğrulama doğruluğu ve farkı kaydet
        trial.set_user_attr("train_r2", train_metrics[-1]["r2"])
        trial.set_user_attr("valid_r2", validation_metrics[-1]["r2"])

        trial.set_user_attr("train_mse", train_metrics[-1]["mse"])
        trial.set_user_attr("valid_mse", validation_metrics[-1]["mse"])

        trial.set_user_attr("train_rmse", train_metrics[-1]["rmse"])
        trial.set_user_attr("valid_rmse", validation_metrics[-1]["rmse"])

        trial.set_user_attr("train_mae", train_metrics[-1]["mae"])
        trial.set_user_attr("valid_mae", validation_metrics[-1]["mae"])
    else:
        trial.set_user_attr("train_r2", -1)
        trial.set_user_attr("valid_r2", -1)

        trial.set_user_attr("train_mse", -1)
        trial.set_user_attr("valid_mse", -1)

        trial.set_user_attr("train_rmse", -1)
        trial.set_user_attr("valid_rmse", -1)

        trial.set_user_attr("train_mae", -1)
        trial.set_user_attr("valid_mae", -1)
        return -1

    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# %%
print(study.best_trial.value, study.best_trial.params)

from deep_metabolitics.config import optuna_optimizers_dir

# %%
from deep_metabolitics.utils.utils import save_pickle

study_df = study.trials_dataframe()
study_df = study_df.sort_values(by="value", ascending=False)
study_df.to_csv(optuna_optimizers_dir / f"{experiment_name}.csv")

save_pickle(data=study, fname=optuna_optimizers_dir / f"{experiment_name}.pickle")
