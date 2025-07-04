import time
from datetime import datetime

import numpy as np
import torch
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deep_metabolitics.data.metabolight_dataset import BaseDataset
from deep_metabolitics.utils.early_stopping import EarlyStopping
from deep_metabolitics.utils.plot import draw_curve
from deep_metabolitics.utils.utils import save_network


def train_one_epoch(model, optimizer, dataloader, logger, fold, epoch):
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


def eval_dataset(model, X, y, logger, fold, running_for, epoch=None):
    start_time = time.time()

    y_pred = model.predict(X)

    end_time = time.time()
    elapsed_time = end_time - start_time

    metrics = {
        "created_at": datetime.now(),
        "fold": fold,
        "running_for": running_for,
        "epoch": epoch,
        "mse": mean_squared_error(y, y_pred, multioutput="uniform_average"),
        "mae": mean_absolute_error(y, y_pred, multioutput="uniform_average"),
        "r2": r2_score(y, y_pred, multioutput="uniform_average"),
        "rmse": root_mean_squared_error(y, y_pred, multioutput="uniform_average"),
        "time": elapsed_time,
        "n_rows": len(X),
    }
    r2_per_output = r2_score(y, y_pred, multioutput="raw_values")
    percentiles = [50, 75, 90, 95, 99, 100]
    for per, per_value in zip(
        percentiles,
        np.percentile(r2_per_output, percentiles),
    ):
        metrics[f"r2_{per}"] = per_value

    logger.info(metrics)
    return metrics


def train(
    train_dataset,
    validation_dataset,
    model,
    logger,
    fold=None,
    fname=None,
):
    start_time = time.time()
    # train_parameter_message = f"learning_rate_{learning_rate}_weight_decay_{weight_decay}_dropout_rate_{dropout_rate}_n_start_layers_{n_start_layers}_batch_size_{batch_size}"
    train_parameter_message = str(model)
    logger.info({"train_parameter_message": train_parameter_message})
    if fold is None:
        fold = train_parameter_message

    X_train = train_dataset.tensors[0].cpu().numpy()
    y_train = train_dataset.tensors[1].cpu().numpy()

    X_val = validation_dataset.tensors[0].cpu().numpy()
    y_val = validation_dataset.tensors[1].cpu().numpy()

    model = model.fit(X_train, y_train)

    # Evaluation Train
    train_metrics = eval_dataset(
        model=model,
        X=X_train,
        y=y_train,
        logger=logger,
        fold=fold,
        running_for="TRAIN",
    )

    # Evaluation Validation
    validation_metrics = eval_dataset(
        model=model,
        X=X_val,
        y=y_val,
        logger=logger,
        fold=fold,
        running_for="VALIDATION",
    )

    if fname is not None:
        save_network(model=model, fname=fname)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info({"trainer_elapsed_time": elapsed_time})
    return model, train_metrics, validation_metrics


def cls_metrics(y_pred, y_true, type):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report
    class_report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(class_report)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("\nAccuracy:", accuracy)

    # Precision
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)

    # Recall
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)

    # F1-Score
    f1 = f1_score(y_true, y_pred)
    print("F1-Score:", f1)

    if len(conf_matrix) == 2:
        metrics = {
            f"{type}_TN": conf_matrix[0][0],
            f"{type}_FP": conf_matrix[0][1],
            f"{type}_FN": conf_matrix[1][0],
            f"{type}_TP": conf_matrix[1][1],
            f"{type}_accuracy": accuracy,
            f"{type}_precision": precision,
            f"{type}_recall": recall,
            f"{type}_f1": f1,
        }
    else:
        metrics = {
            f"{type}_TN": 0,
            f"{type}_FP": 0,
            f"{type}_FN": 0,
            f"{type}_TP": 0,
            f"{type}_accuracy": accuracy,
            f"{type}_precision": precision,
            f"{type}_recall": recall,
            f"{type}_f1": f1,
        }

    return metrics


def cls_results(X_train, y_train, X_test, y_test):
    metrics = {}
    # classification_pipeline = Pipeline(
    #     [
    #         ("pca", PCA()),
    #         ("std", StandardScaler()),
    #         ("clf", LogisticRegression(C=0.3e-11, random_state=43)),
    #     ]
    # )
    from sklearn.ensemble import RandomForestClassifier

    # classification_pipeline = Pipeline(
    #     [
    #         # ("vect", DictVectorizer(sparse=False)),
    #         ("pca", PCA()),
    #         ("clf", LogisticRegression(C=0.3e-6, random_state=43)),
    #     ]
    # )
    classification_pipeline = Pipeline(
        [
            # ("vect", DictVectorizer(sparse=False)),
            ("pca", PCA(random_state=43)),
            ("clf", RandomForestClassifier(random_state=43)),
        ]
    )
    classification_pipeline.fit(X_train, y_train)
    y_train_pred = classification_pipeline.predict(X_train)
    train_metrics = cls_metrics(y_pred=y_train_pred, y_true=y_train, type="train")
    metrics.update(train_metrics)

    y_test_pred = classification_pipeline.predict(X_test)
    test_metrics = cls_metrics(y_pred=y_test_pred, y_true=y_test, type="test")
    metrics.update(test_metrics)
    return metrics


def classification_evaluation(
    model,
    train_dataset,
    test_dataset,
    dataset_name,
    logger,
    fold,
    running_for="classification",
):
    X_train, train_pathway_true, train_y = train_dataset
    X_test, test_pathway_true, test_y = test_dataset

    train_pathway_pred = model.predict(X_train)
    test_pathway_pred = model.predict(X_test)

    metrics = {}

    true_metrics = cls_results(
        X_train=train_pathway_true,
        y_train=train_y,
        X_test=test_pathway_true,
        y_test=test_y,
    )
    true_metrics = {f"true_{key}": value for key, value in true_metrics.items()}
    metrics.update(true_metrics)

    pred_metrics = cls_results(
        X_train=train_pathway_pred,
        y_train=train_y,
        X_test=test_pathway_pred,
        y_test=test_y,
    )
    pred_metrics = {f"pred_{key}": value for key, value in pred_metrics.items()}
    metrics.update(pred_metrics)

    metrics["dataset_name"] = dataset_name
    metrics["fold"] = fold
    metrics["running_for"] = running_for
    metrics["len_train"] = len(train_pathway_true)
    metrics["len_test"] = len(test_pathway_true)

    logger.info(metrics)
    return metrics
