import time
from datetime import datetime

import numpy as np
import torch
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
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


def predict(model, dataset):
    if dataset is not None:
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        test_batch_size = 64
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size)
        # dataloader = torch.utils.data.DataLoader(dataset)
        model.eval()

        # factor_list = []
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for X, y_true in dataloader:
                y_pred = model(X)
                if len(y_true) > 1:
                    # factor_list.extend(factor.squeeze().cpu().tolist())
                    y_true_list.extend(y_true.squeeze().cpu().tolist())
                    y_pred_list.extend(y_pred.squeeze().cpu().tolist())
                else:
                    # factor_list.append(factor.squeeze().cpu().tolist())
                    y_true_list.append(y_true.squeeze().cpu().tolist())
                    y_pred_list.append(y_pred.squeeze().cpu().tolist())

            # factor_list = np.array(factor_list)
            y_true_list = np.array(y_true_list)
            y_pred_list = np.array(y_pred_list)
    return y_true_list, y_pred_list


def eval_dataset(model, dataset, logger, fold, running_for, epoch=None):
    start_time = time.time()
    y_true_list, y_pred_list = predict(model, dataset)

    end_time = time.time()
    elapsed_time = end_time - start_time

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
        "mape": mean_absolute_percentage_error(
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

    q_errors = q_error_function(y_true_list, y_pred_list)
    for per, per_value in zip(
        percentiles,
        np.percentile(q_errors, percentiles),
    ):
        metrics[f"qerror_{per}"] = per_value
    logger.info(metrics)
    return metrics


def q_error_function(y_true, y_pred):
    """
    Q-error (quotient error) hesaplar.

    Parameters:
    -----------
    y_true : numpy.ndarray
        Gerçek değerler
    y_pred : numpy.ndarray
        Tahmin edilen değerler

    Returns:
    --------
    float or numpy.ndarray
        Her bir örnek için q-error değerleri veya ortalama q-error

    Notes:
    ------
    Q-error = max(tahmin/gerçek, gerçek/tahmin)
    """
    # Sıfıra bölme hatalarını önlemek için küçük bir epsilon değeri
    eps = 5

    # Girişleri numpy array'e çevir
    y_true = np.abs(np.asarray(y_true)) + eps
    y_pred = np.abs(np.asarray(y_pred)) + eps

    # Q-error hesaplama
    ratio1 = y_pred / y_true
    ratio2 = y_true / y_pred

    # Her örnek için maksimum oranı al
    q_errors = np.abs(np.maximum(ratio1, ratio2))

    return q_errors


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

    early_stopping = EarlyStopping(
        patience=early_stopping_patience, min_delta=early_stopping_min_delta
    )

    train_metrics_list = []
    validation_metrics_list = []

    for epoch in range(epochs):
        model, optimizer = train_one_epoch(
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
    from xgboost import XGBClassifier

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
            # ("pca", PCA(random_state=43)),
            ("clf", XGBClassifier(random_state=43)),
        ]
    )
    classification_pipeline.fit(X_train, y_train)
    y_train_pred = classification_pipeline.predict(X_train)
    train_metrics = cls_metrics(y_pred=y_train_pred, y_true=y_train, type="train")
    metrics.update(train_metrics)

    y_test_pred = classification_pipeline.predict(X_test)
    test_metrics = cls_metrics(y_pred=y_test_pred, y_true=y_test, type="test")
    metrics.update(test_metrics)
    
    
    xgb_model = classification_pipeline.named_steps["clf"]
    importances = xgb_model.feature_importances_
    feature_names = X_train.columns
    import shap
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_train)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    
    
    import pandas as pd

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "shap_importance": shap_importance
    }).sort_values(by="importance", ascending=False)
    return metrics, feature_importance_df


def classification_evaluation(
    model,
    train_dataset,
    test_dataset,
    dataset_name,
    logger,
    fold,
    running_for="classification",
):
    metrics = {}
    train_pathway_true, train_pathway_pred, train_y = predict(
        model=model, dataset=train_dataset
    )
    test_pathway_true, test_pathway_pred, test_y = predict(
        model=model, dataset=test_dataset
    )

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
