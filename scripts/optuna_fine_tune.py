import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import random
import time

import torch
from sklearn.model_selection import KFold

from deep_metabolitics.config import logs_dir, models_dir
from deep_metabolitics.data.metabolight_dataset import PathwayDataset
from deep_metabolitics.data.properties import get_dataset_ids
from deep_metabolitics.utils.early_stopping import EarlyStopping
from deep_metabolitics.utils.logger import create_logger
from deep_metabolitics.utils.plot import draw_curve
from deep_metabolitics.utils.trainer import eval_dataset, train_one_epoch
from deep_metabolitics.utils.utils import get_device, save_network

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


VALIDATION_RATE = 0.2

# BATCH_SIZE = 8 # 8
SHUFFLE = True
# LR = 1e-2

# epochs = 100

experiment_name = "fine_tune_324"
logger = create_logger(experiment_name)

dataset_ids = get_dataset_ids()

train_index = [3]
# test_index = None

import torch

from deep_metabolitics.utils.utils import get_device


class MultioutRegressorNET(torch.nn.Module):
    def __init__(
        self,
        n_features: int,  # of metabolights
        out_features: int,  # of pathways
        n_start_layers: int,
        dropout_rate: float,
        loss_method: str = "MSE",
    ):
        """
        Parameters:
            n_features (int): Initial feature size of the specific study, columns
        """
        super(MultioutRegressorNET, self).__init__()

        self.n_features = n_features
        self.out_features = out_features
        self.n_start_layers = n_start_layers
        self.dropout_rate = dropout_rate
        self.loss_method = loss_method

        start_layers = []
        input_size = self.n_features
        for i in range(self.n_start_layers):
            output_size = int(self.n_features / 2 + i)

            start_layers.append(torch.nn.Linear(input_size, output_size))
            start_layers.append(torch.nn.BatchNorm1d(output_size))
            start_layers.append(torch.nn.ReLU())
            start_layers.append(torch.nn.Dropout(p=self.dropout_rate))

            input_size = output_size
        output_size = int(self.out_features / 2)
        end_layers = [
            torch.nn.Linear(input_size, output_size),
            torch.nn.BatchNorm1d(output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_rate),
            torch.nn.Linear(output_size, self.out_features),
        ]

        layers = start_layers + end_layers
        self.model = torch.nn.Sequential(*layers)

        self.device = self.__get_device()
        self.to(device=self.device)

    @staticmethod
    def __get_device():
        device = get_device()
        print(f"PyTorch: Training model on device {device}.")
        return device

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        x = self.model(x)

        return x

    def loss_function(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        if self.loss_method == "MSE":
            loss = torch.nn.MSELoss(reduction="mean")(y_true, y_pred)
        # elif self.loss_method == "RMSE":
        #     loss = torch.nn.RMSELoss(y_true, y_pred)  # RMSE
        elif self.loss_method == "MAE":
            loss = torch.nn.L1Loss(reduction="mean")(y_true, y_pred)  # MAE
        else:
            raise Exception("INVALID LOSS METHOD")

        return loss


def train(
    epochs, learning_rate, weight_decay, dropout_rate, n_start_layers, batch_size
):
    tune_name = f"learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}"
    logger.info({"tune_info": tune_name})

    dataset = PathwayDataset(dataset_ids=dataset_ids[train_index])

    n_features = dataset.n_metabolights
    out_features = dataset.n_labels

    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [1 - VALIDATION_RATE, VALIDATION_RATE]
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(batch_size), shuffle=SHUFFLE
    )

    model = MultioutRegressorNET(
        n_features=n_features,
        out_features=out_features,
        n_start_layers=n_start_layers,
        dropout_rate=dropout_rate,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(epochs / 5), gamma=0.1
    )

    train_r2 = []
    validation_r2 = []

    train_loss = []
    validation_loss = []

    for epoch in range(epochs):
        model, optimizer = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            logger=logger,
            fold=tune_name,
            epoch=epoch,
        )

        # Evaluation Train
        train_metrics = eval_dataset(
            model=model,
            dataset=train_dataset,
            logger=logger,
            fold=tune_name,
            running_for="TRAIN",
            epoch=epoch,
        )
        train_r2.append(train_metrics["r2"])
        train_loss.append(train_metrics["loss"])

        # Evaluation Validation
        validation_metrics = eval_dataset(
            model=model,
            dataset=validation_dataset,
            logger=logger,
            fold=None,
            running_for="VALIDATION",
            epoch=epoch,
        )

        validation_r2.append(validation_metrics["r2"])
        validation_loss.append(validation_metrics["loss"])
        scheduler.step()

    draw_curve(
        epochs=epochs,
        train_values=train_r2,
        val_values=validation_r2,
        metric_name=f"{tune_name} R2",
    )
    draw_curve(
        epochs=epochs,
        train_values=train_loss,
        val_values=validation_loss,
        metric_name=f"{tune_name} Loss",
    )
    return max(validation_r2)


import optuna


def objective(trial):
    epochs = 200

    # learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    learning_rate = trial.suggest_discrete_uniform(
        "learning_rate", 1e-3, 1e-1, 0.1
    )  # 3
    # learning_rate = 0.1
    weight_decay = trial.suggest_discrete_uniform("weight_decay", 1e-5, 1e-1, 0.2)  # 3
    # weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    # weight_decay = 0.0004
    dropout_rate = trial.suggest_discrete_uniform("dropout_rate", 0.1, 0.9, 0.3)  # 3
    # dropout_rate = 0.4
    n_start_layers = trial.suggest_int("n_start_layers", 1, 10, 3)  # 4
    # batch_size = trial.suggest_int("batch_size", 8, 64, 8) # 8
    batch_size = trial.suggest_categorical("batch_size", [8, 32, 64])  # 3

    # Modeli oluştur ve eğit
    r2 = train(
        epochs, learning_rate, weight_decay, dropout_rate, n_start_layers, batch_size
    )
    return r2


optuna.logging.enable_propagation()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3 * 3 * 3 * 4 * 3)
optuna.logging.disable_propagation()
