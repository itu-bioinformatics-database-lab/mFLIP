import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from torch_geometric.loader import DataLoader

from deep_metabolitics.data.metabolight_dataset import BaseDataset
from deep_metabolitics.utils.early_stopping import EarlyStopping
from deep_metabolitics.utils.plot import draw_curve
from deep_metabolitics.utils.utils import save_network


def train_one_epoch(model, optimizer, batch):
    epoch_loss = 0
    # Iterate over mini batches
    model.train()
    X, y = batch
    optimizer.zero_grad()
    # Forwad Pass
    y_pred = model(X)

    # Loss Calculation
    batch_loss = model.loss_function(y_true=y, y_pred=y_pred)

    # Backward Pass

    batch_loss.backward()

    # Parameter update
    optimizer.step()

    # Update epoch loss
    epoch_loss = batch_loss.item()

    return model, optimizer, epoch_loss


def predict_all(dataset, model):
    start_time = time.time()
    y_pred_list = None
    y_true_list = None
    elapsed_time = None
    if dataset is not None:
        # dataloader = DataLoader(dataset, batch_size=1)
        model.eval()

        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for X, y_true in dataset:
                y_pred = model(X)
                y_true_list.append(y_true.squeeze().cpu().tolist())
                y_pred_list.append(y_pred.squeeze().cpu().tolist())

        y_true_list = np.array(y_true_list)
        y_pred_list = np.array(y_pred_list)

        end_time = time.time()
        elapsed_time = end_time - start_time

    return y_pred_list, y_true_list, elapsed_time


def eval_dataset(model, dataset, logger, fold, running_for, epoch=None):
    y_pred_list, y_true_list, elapsed_time = predict_all(dataset, model)
    if elapsed_time is not None:
        metrics = {
            "created_at": datetime.now(),
            "fold": fold,
            "running_for": running_for,
            "epoch": epoch,
            "mse": mean_squared_error(
                y_true_list, y_pred_list, multioutput="uniform_average"
            ),
            "mae": mean_absolute_error(
                y_true_list, y_pred_list, multioutput="uniform_average"
            ),
            "r2": r2_score(y_true_list, y_pred_list, multioutput="uniform_average"),
            "rmse": root_mean_squared_error(
                y_true_list, y_pred_list, multioutput="uniform_average"
            ),
            "time": elapsed_time,
            "n_rows": len(dataset),
        }
        r2_per_output = r2_score(y_true_list, y_pred_list, multioutput="raw_values")
        percentiles = [50, 75, 90, 95, 99, 100]
        for per, per_value in zip(
            percentiles,
            np.percentile(r2_per_output, percentiles),
        ):
            metrics[f"r2_{per}"] = per_value

        logger.info(metrics)
        return metrics
    return {}


def train(
    epochs,
    dataloader,
    train_dataset,
    validation_dataset,
    model,
    # optimizer,
    # scheduler,
    learning_rate,
    weight_decay,
    dropout_rate,
    n_start_layers,
    batch_size,
    logger,
    scheduler_step_size,
    scheduler_gamma,
    early_stopping_patience=20,
    early_stopping_min_delta=0.001,
    early_stopping_metric_name="mse",
    fold=None,
    fname=None,
):
    start_time = time.time()
    # train_parameter_message = f"learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}"
    train_parameter_message = f"epochs_{epochs}_learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}_scheduler_step_size_{scheduler_step_size}_scheduler_gamma_{scheduler_gamma}_early_stopping_patience_{early_stopping_patience}_early_stopping_min_delta_{early_stopping_min_delta}_early_stopping_metric_name_{early_stopping_metric_name}"
    logger.info({"train_parameter_message": train_parameter_message})
    if fold is None:
        fold = train_parameter_message

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )

    for _ in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch_loss = 0
            for _ in range(1):
                model, optimizer, epoch_loss = train_one_epoch(model, optimizer, batch)
                batch_loss += epoch_loss
            batch_loss /= epochs
            epoch_loss += batch_loss
        epoch_loss /= epochs
        logger.info({"epoch_loss": epoch_loss})
        scheduler.step()

    if fname is not None:
        save_network(model=model, fname=fname)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info({"trainer_elapsed_time": elapsed_time})
    return model, optimizer


def train_only(
    epochs,
    dataloader,
    model,
    learning_rate,
    weight_decay,
    dropout_rate,
    n_start_layers,
    batch_size,
    logger,
    scheduler_step_size,
    scheduler_gamma,
    early_stopping_patience=20,
    early_stopping_min_delta=0.001,
    early_stopping_metric_name="mse",
    fold=None,
    fname=None,
):
    start_time = time.time()
    # train_parameter_message = f"learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}"
    train_parameter_message = f"epochs_{epochs}_learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}_scheduler_step_size_{scheduler_step_size}_scheduler_gamma_{scheduler_gamma}_early_stopping_patience_{early_stopping_patience}_early_stopping_min_delta_{early_stopping_min_delta}_early_stopping_metric_name_{early_stopping_metric_name}"
    logger.info({"train_parameter_message": train_parameter_message})
    if fold is None:
        fold = train_parameter_message

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    for batch in dataloader:
        batch_loss = 0
        for epoch in range(epochs):
            model, optimizer, epoch_loss = train_one_epoch(model, optimizer, batch)
            batch_loss += epoch_loss
        batch_loss /= epochs
        logger.info({"batch_loss": batch_loss})

    if fname is not None:
        save_network(model=model, fname=fname)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info({"trainer_elapsed_time": elapsed_time})
    return model, optimizer


def train_one_epoch_orji(model, optimizer, dataloader, logger, fold, epoch):
    epoch_loss = 0
    # Iterate over mini batches
    model.train()
    for x_train_batch, y_train_batch in dataloader:
        optimizer.zero_grad()
        # Forwad Pass
        y_pred = model(x_train_batch)

        # Loss Calculation
        batch_loss = model.loss_function(y_true=y_train_batch, y_pred=y_pred)

        # Backward Pass

        batch_loss.backward()

        # Parameter update
        optimizer.step()

        # Update epoch loss
        epoch_loss += batch_loss.item()

    # One epoch of training complete, calculate average training epoch loss
    epoch_loss /= len(dataloader)
    logger.info(
        {"fold": fold, "running_for": "TRAIN", "epoch": epoch, "loss": epoch_loss}
    )
    return model, optimizer


def train_orji(
    epochs,
    dataloader,
    train_dataset,
    validation_dataset,
    model,
    # optimizer,
    # scheduler,
    learning_rate,
    weight_decay,
    dropout_rate,
    n_start_layers,
    batch_size,
    logger,
    scheduler_step_size,
    scheduler_gamma,
    early_stopping_patience=20,
    early_stopping_min_delta=0.001,
    early_stopping_metric_name="mse",
    fold=None,
    fname=None,
):
    start_time = time.time()
    # train_parameter_message = f"learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}"
    train_parameter_message = f"epochs_{epochs}_learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}_scheduler_step_size_{scheduler_step_size}_scheduler_gamma_{scheduler_gamma}_early_stopping_patience_{early_stopping_patience}_early_stopping_min_delta_{early_stopping_min_delta}_early_stopping_metric_name_{early_stopping_metric_name}"
    logger.info({"train_parameter_message": train_parameter_message})
    if fold is None:
        fold = train_parameter_message

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )

    early_stopping = EarlyStopping(
        patience=early_stopping_patience, min_delta=early_stopping_min_delta
    )

    train_metrics_list = []
    validation_metrics_list = []

    for epoch in range(epochs):
        model, optimizer = train_one_epoch_orji(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            logger=logger,
            fold=fold,
            epoch=epoch,
        )

        # Evaluation Train
        train_metrics = eval_dataset(
            model=model,
            dataset=train_dataset,
            logger=logger,
            fold=fold,
            running_for="TRAIN",
            epoch=epoch,
        )
        train_metrics_list.append(train_metrics)

        # Evaluation Validation
        validation_metrics = eval_dataset(
            model=model,
            dataset=validation_dataset,
            logger=logger,
            fold=fold,
            running_for="VALIDATION",
            epoch=epoch,
        )

        validation_metrics_list.append(validation_metrics)
        scheduler.step()
        # Early stop check
        early_stopping_metric = validation_metrics[
            early_stopping_metric_name
        ]  # TODO we may send r2
        early_stopping(early_stopping_metric)
        if early_stopping.early_stop:
            logger.info(
                {
                    "fold": fold,
                    "message_for": "early_stopping",
                    "epoch": epoch,
                    "last_validation_loss": early_stopping_metric,
                }
            )
            break

    draw_curve(
        epochs=epochs,
        train_values=[metrics["r2"] for metrics in train_metrics_list],
        val_values=[metrics["r2"] for metrics in validation_metrics_list],
        metric_name=f"{train_parameter_message} R2",
    )
    draw_curve(
        epochs=epochs,
        train_values=[metrics["mse"] for metrics in train_metrics_list],
        val_values=[metrics["mse"] for metrics in validation_metrics_list],
        metric_name=f"{train_parameter_message} MSE",
    )
    if fname is not None:
        save_network(model=model, fname=fname)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info({"trainer_elapsed_time": elapsed_time})
    return model, optimizer, train_metrics_list, validation_metrics_list
