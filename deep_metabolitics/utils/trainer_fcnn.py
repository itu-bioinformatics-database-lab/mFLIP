import time

import numpy as np
import torch
from sklearn.metrics import (
    d2_absolute_error_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
from tqdm import tqdm


def tensor_to_numpy(dataset):
    metabolites = np.array(
        [dataset[i][0].detach().cpu().numpy() for i in range(len(dataset))]
    )
    pathways = np.array(
        [dataset[i][1].detach().cpu().numpy() for i in range(len(dataset))]
    )
    return metabolites, pathways


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


def train_sklearn(
    train_dataset,
    validation_dataset,
    model,
    logger=None,
    fold=None,
    fname=None,
    pathway_names=None,
):
    X_train, y_train = tensor_to_numpy(train_dataset)
    # X_val, y_val = tensor_to_numpy(validation_dataset)

    start_time = time.time()

    model = model.fit(X_train, y_train)
    end_time = time.time()
    train_elapsed_time = end_time - start_time

    # Evaluation Train
    train_metrics = evaluate(
        model=model,
        dataset=train_dataset,
        pathway_names=pathway_names,
        device=None,
        threshold=2,
        scaler=None,
    )

    train_metrics["train_fit_elapsed_time"] = train_elapsed_time

    # Evaluation Validation
    validation_metrics = evaluate(
        model=model,
        dataset=validation_dataset,
        pathway_names=pathway_names,
        device=None,
        threshold=2,
        scaler=None,
    )

    return model, train_metrics, validation_metrics


def train(
    epochs,
    dataloader,
    train_dataset,
    validation_dataset,
    model,
    learning_rate=0.0001,
    weight_decay=0.01,
    scheduler_step_size=50,
    scheduler_gamma=0.8,
    early_stopping_patience=100,
    early_stopping_min_delta=0.0001,
    logger=None,
    fold=None,
    pathway_names=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    print_every=10,
):
    model = model.to(device)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Early stopping setup
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    train_metrics_list = []
    validation_metrics_list = []

    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        # Training phase
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_l1_loss = 0
        total_huber_loss = 0
        total_correlation_loss = 0

        for batch in dataloader:
            metabolites, pathway_targets = batch
            metabolites = metabolites.to(device)
            pathway_targets = pathway_targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(metabolites)

            # Calculate losses
            loss_dict = model.loss_function(predictions, pathway_targets)

            # Backward pass
            loss_dict["loss"].backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0)
            optimizer.step()

            total_loss += loss_dict["loss"].item()
            total_mse_loss += loss_dict["mse_loss"].item()
            total_l1_loss += loss_dict["l1_loss"].item()
            total_huber_loss += loss_dict["huber_loss"].item()
            # total_correlation_loss += loss_dict["correlation_loss"].item()
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Calculate average losses
        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_mse_loss = total_mse_loss / n_batches
        avg_l1_loss = total_l1_loss / n_batches
        avg_huber_loss = total_huber_loss / n_batches
        # avg_correlation_loss = total_correlation_loss / n_batches

        if logger:
            logger.info(
                {
                    "fold": fold,
                    "running_for": "TRAIN",
                    "epoch": epoch,
                    "loss": avg_loss,
                    "mse_loss": avg_mse_loss,
                    "l1_loss": avg_l1_loss,
                    "huber_loss": avg_huber_loss,
                    # "correlation_loss": avg_correlation_loss,
                }
            )

        # Train evaluation phase
        model.eval()
        train_metrics = evaluate(
            model=model,
            dataset=train_dataset,
            device=device,
            pathway_names=pathway_names,
        )
        train_metrics["train_fit_elapsed_time"] = elapsed_time
        # Validation phase
        model.eval()
        val_metrics = evaluate(
            model=model,
            dataset=validation_dataset,
            device=device,
            pathway_names=pathway_names,
        )
        train_metrics_list.append(train_metrics)
        validation_metrics_list.append(val_metrics)

        # if epoch % print_every == 0:
        #     print(
        #         f"Epoch {epoch} Train metrics loss: r2_stats: {train_metrics['r2_stats']} q_error_stats: {train_metrics['q_error_stats']} loss: {train_metrics['loss']} mse: {train_metrics['mse']} r2: {train_metrics['r2']}"
        #     )
        #     print(
        #         f"Epoch {epoch} Validation metrics loss: r2_stats: {val_metrics['r2_stats']} q_error_stats: {val_metrics['q_error_stats']} loss: {val_metrics['loss']} mse: {val_metrics['mse']} r2: {val_metrics['r2']}"
        #     )

        if logger:
            logger.info(
                {
                    "fold": fold,
                    "running_for": "TRAIN",
                    "epoch": epoch,
                    **train_metrics,
                }
            )
            logger.info(
                {
                    "fold": fold,
                    "running_for": "VALIDATION",
                    "epoch": epoch,
                    **val_metrics,
                }
            )

        # Early stopping check
        current_val_loss = val_metrics["loss"]
        if current_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if logger:
                logger.info(
                    f"EarlyStopping Counter: {patience_counter} out of {early_stopping_patience}"
                )

        if patience_counter >= early_stopping_patience:
            if logger:
                logger.info("Early stopping triggered")
            break

        # Step the scheduler with validation loss
        scheduler.step(current_val_loss)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, optimizer, train_metrics_list, validation_metrics_list


def predict_deep(model, dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    total_loss = 0
    total_mse_loss = 0
    total_l1_loss = 0
    total_huber_loss = 0
    total_correlation_loss = 0

    all_predictions = []
    all_targets = []
    start_time = time.time()
    with torch.no_grad():
        for datapoint in dataloader:
            metabolites, pathway_targets = datapoint
            metabolites = metabolites.to(device)
            pathway_targets = pathway_targets.to(device)

            # Forward pass
            if len(metabolites.shape) < 2:
                metabolites = metabolites.unsqueeze(0)
                pathway_targets = pathway_targets.unsqueeze(0)
            predictions = model(metabolites)

            # Calculate losses
            loss_dict = model.loss_function(predictions, pathway_targets)
            total_loss += loss_dict["loss"].item()
            total_mse_loss += loss_dict["mse_loss"].item()
            total_l1_loss += loss_dict["l1_loss"].item()
            total_huber_loss += loss_dict["huber_loss"].item()
            # total_correlation_loss += loss_dict["correlation_loss"].item()
            predictions = predictions["pathways_pred"]
            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(pathway_targets.cpu().numpy())
    end_time = time.time()
    # Calculate average losses
    n_samples = len(dataset)
    avg_loss = total_loss / n_samples
    avg_mse_loss = total_mse_loss / n_samples
    avg_huber_loss = total_huber_loss / n_samples
    # avg_correlation_loss = total_correlation_loss / n_samples

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    metrics = {
        "predict_time": end_time - start_time,
        "n_samples": len(dataset),
        "loss": avg_loss,
        "mse_loss": avg_mse_loss,
        "huber_loss": avg_huber_loss,
    }
    return all_predictions, all_targets, metrics


def predict_sklearn(model, dataset):
    X, y = tensor_to_numpy(dataset)
    start_time = time.time()
    all_predictions = model.predict(X)
    end_time = time.time()
    metrics = {
        "predict_time": end_time - start_time,
        "n_samples": len(dataset),
    }
    return all_predictions, y, metrics


def calculate_metric_stats(metrics, metric_name):
    metric_scores = [metric[metric_name] for metric in metrics.values()]

    # İstatistikleri hesaplama
    metric_stats = {
        f"{metric_name}_mean": np.mean(metric_scores),
        f"{metric_name}_q25": np.percentile(metric_scores, 25),
        f"{metric_name}_q50": np.percentile(metric_scores, 50),  # medyan
        f"{metric_name}_q75": np.percentile(metric_scores, 75),
        f"{metric_name}_q90": np.percentile(metric_scores, 90),
        f"{metric_name}_q99": np.percentile(metric_scores, 99),
        f"{metric_name}_q100": np.max(metric_scores),  # maksimum değer
    }
    return metric_stats


def evaluate(
    model,
    dataset,
    pathway_names,
    device="cuda" if torch.cuda.is_available() else "cpu",
    threshold=5,
    scaler=None,
):
    if device is not None:
        all_predictions, all_targets, metrics = predict_deep(
            model=model,
            dataset=dataset,
            device=device,
        )
    else:
        all_predictions, all_targets, metrics = predict_sklearn(
            model=model,
            dataset=dataset,
        )
    n_pathways = all_targets.shape[1]
    pathway_metrics = {}
    for i in range(n_pathways):
        pathway_name = pathway_names[i]
        #     q_error = q_error_function(all_targets[:, i], all_predictions[:, i])
        #     q_error = np.median(q_error)
        # pathway_metrics[pathway_name] = {"q_error": q_error}
        pathway_metrics[pathway_name] = {}

    if scaler is not None:
        all_predictions = scaler.inverse_transform(all_predictions)
        all_targets = scaler.inverse_transform(all_targets)
        mask = (all_predictions >= -threshold) & (
            all_predictions <= threshold
        )  # -2 ile 2 arasındaki değerleri bul
        all_predictions[mask] = 0  # Sadece bu aralıktaki değerleri 0'a çevir

    # Calculate metrics for each pathway
    # pathway_metrics = []

    for i in range(n_pathways):
        pathway_name = pathway_names[i]
        pathway_metrics[pathway_name].update(
            {
                "pathway_idx": i,
                "pathway_name": pathway_name,
                "actual": all_targets[:, i],
                "predicted": all_predictions[:, i],
                # "r2": r2_score(all_targets[:, i], all_predictions[:, i]),
                # "explained_variance": explained_variance_score(
                #     all_targets[:, i], all_predictions[:, i]
                # ),
                # "rmse": np.sqrt(
                #     mean_squared_error(all_targets[:, i], all_predictions[:, i])
                # ),
                # "mse": mean_squared_error(all_targets[:, i], all_predictions[:, i]),
                # "mae": mean_absolute_error(all_targets[:, i], all_predictions[:, i]),
                # "d2_absolute_error": d2_absolute_error_score(
                #     all_targets[:, i], all_predictions[:, i]
                # ),
                # "max_error": max_error(all_targets[:, i], all_predictions[:, i]),
                # "mean_absolute_percentage_error": mean_absolute_percentage_error(
                #     all_targets[:, i], all_predictions[:, i]
                # ),
                # "mean_squared_log_error": mean_squared_log_error(
                #     all_targets[:, i], all_predictions[:, i]
                # ),
                # "median_absolute_error": median_absolute_error(
                #     all_targets[:, i], all_predictions[:, i]
                # ),
                # "mean_poisson_deviance": mean_poisson_deviance(
                #     all_targets[:, i], all_predictions[:, i]
                # ),
                # "mean_gamma_deviance": mean_gamma_deviance(
                #     all_targets[:, i], all_predictions[:, i]
                # ),
            }
        )

    # r2_stats = calculate_metric_stats(pathway_metrics, "r2")
    # q_error_stats = calculate_metric_stats(pathway_metrics, "q_error")
    # explained_variance_stats = calculate_metric_stats(
    #     pathway_metrics, "explained_variance"
    # )

    metrics["mse"] = mean_squared_error(all_targets, all_predictions)
    metrics["mae"] = mean_absolute_error(all_targets, all_predictions)
    metrics["rmse"] = np.sqrt(mean_squared_error(all_targets, all_predictions))
    metrics["r2"] = r2_score(all_targets, all_predictions)
    metrics["pathway_metrics"] = pathway_metrics
    # metrics["r2_stats"] = r2_stats
    # metrics["q_error_stats"] = q_error_stats
    # metrics["explained_variance_stats"] = explained_variance_stats
    # classification_metrics = get_classification_metrics(
    #     pathway_metrics=pathway_metrics.values(), threshold=threshold
    # )  # TODO burada farkli seyler olacak
    # metrics["classification_metrics"] = classification_metrics
    return metrics


def plot_pathway_predictions(
    pathway_metrics, source, save_dir=None, cols=3, figsize=(20, 20)
):
    """
    Plot actual vs predicted values for all pathways in a single figure

    Args:
        pathway_metrics: List of dictionaries containing metrics for each pathway
        save_dir: Directory to save the plots (optional)
        cols: Number of columns in the subplot grid
        figsize: Figure size as (width, height)
    """
    import math

    import matplotlib.pyplot as plt
    import seaborn as sns

    n_pathways = len(pathway_metrics)
    rows = math.ceil(n_pathways / cols)

    fig = plt.figure(figsize=figsize)

    for idx, metric in enumerate(pathway_metrics, 1):
        ax = fig.add_subplot(rows, cols, idx)

        # Create scatter plot
        sns.scatterplot(x=metric["actual"], y=metric["predicted"], alpha=0.5, ax=ax)

        # Add diagonal line (perfect prediction line)
        min_val = min(metric["actual"].min(), metric["predicted"].min())
        max_val = max(metric["actual"].max(), metric["predicted"].max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction"
        )

        # Add horizontal and vertical lines at y=0 and x=0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Add metrics text
        text = f"R² = {metric['r2']:.3f}\nRMSE = {metric['rmse']:.3f}\nMAE = {metric['mae']:.3f}"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Labels and title
        pathway_name = metric["pathway_name"]
        ax.set_title(f"{pathway_name}", fontsize=10)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save or show plot
    if save_dir:
        plt.savefig(
            f"{save_dir}/{source}_all_pathways_predictions.png",
            dpi=300,
            bbox_inches="tight",
        )
        # plt.close()
        plt.show()
    else:
        plt.show()


def convert_to_classes(values, threshold=2):
    """Convert continuous values to -1, 0, 1 based on thresholds"""
    classes = np.zeros_like(values)
    classes[values > threshold] = 1
    classes[values < -threshold] = -1
    return classes


def get_classification_metrics(pathway_metrics, threshold=2):
    """
    Plot actual vs predicted values and confusion matrices for all pathways

    Args:
        pathway_metrics: List of dictionaries containing metrics for each pathway
        save_dir: Directory to save the plots (optional)
        cols: Number of columns in the subplot grid
        figsize: Figure size as (width, height)
        threshold: Threshold for classification (-threshold < 0 < threshold)
    """

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    # Return classification metrics for all pathways
    classification_metrics = {}
    for metric in pathway_metrics:
        actual_classes = convert_to_classes(metric["actual"], threshold)
        predicted_classes = convert_to_classes(metric["predicted"], threshold)
        class_report = classification_report(
            actual_classes, predicted_classes, output_dict=True
        )

        classification_metrics[metric["pathway_name"]] = {
            "accuracy": class_report["accuracy"],
            "f1_micro": f1_score(
                actual_classes, predicted_classes, average="micro", zero_division=0
            ),
            "f1_macro": f1_score(
                actual_classes, predicted_classes, average="macro", zero_division=0
            ),
            "f1_weighted": f1_score(
                actual_classes, predicted_classes, average="weighted", zero_division=0
            ),
            "class_report": class_report,
            "confusion_matrix": confusion_matrix(
                actual_classes, predicted_classes, normalize="true", labels=[-1, 0, 1]
            ),
        }

    return {"pathway_metrics": classification_metrics}


def warmup_training(model, train_dataset, num_warmup_steps=1000):
    from torch.utils.data import DataLoader
    batch_size = 32
    # Küçük learning rate ile başla
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size #, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )  # TODO burasini optimize edelim
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    for step in range(num_warmup_steps):
        for batch in train_loader:
            metabolites, pathway_targets = batch
            metabolites = metabolites.to(device)
            pathway_targets = pathway_targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(metabolites)

            # Calculate losses
            loss_dict = model.loss_function(predictions, pathway_targets)

            # Backward pass
            loss_dict["loss"].backward()
            optimizer.step()
    return model
