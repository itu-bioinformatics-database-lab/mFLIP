import time

import numpy as np
import torch
from tqdm import tqdm


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
    print_every=1,
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
        total_mse_loss = 0
        total_loss = 0

        train_data_loss_list = []

        for batch in dataloader:
            metabolites, targets = batch

            for _ in range(10):
                optimizer.zero_grad()

                # Forward pass
                predictions = model(metabolites)

                # Calculate losses
                loss_dict = model.loss_function(predictions, targets)

                # Backward pass
                loss_dict["loss"].backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_mse_loss += loss_dict["mse_loss"].item()
            total_loss += loss_dict["loss"].item()
            train_data_loss_list.append(loss_dict)

        end_time = time.time()
        elapsed_time = end_time - start_time
        # Calculate average losses
        n_batches = len(dataloader)
        avg_mse_loss = total_mse_loss / n_batches
        total_loss = total_loss / n_batches

        train_metrics_list.append(
            {
                "mse_loss": avg_mse_loss,
                "data_loss_list": train_data_loss_list,
                "elapsed_time": elapsed_time,
            }
        )

        # Train evaluation phase
        model.eval()
        validation_avg_mseloss = 0
        validation_avg_loss = 0
        validation_data_loss_list = []
        for batch in validation_dataset:
            data, targets = batch
            model.eval()
            with torch.no_grad():
                predictions = model(data)
                # MSE Loss
                loss_dict = model.loss_function(predictions, targets)

                validation_avg_mseloss += loss_dict["mse_loss"].item()
                validation_avg_loss += loss_dict["loss"].item()
                validation_data_loss_list.append(loss_dict)

        validation_avg_mseloss = validation_avg_mseloss / len(validation_dataset)
        validation_avg_loss = validation_avg_loss / len(validation_dataset)
        validation_metrics_list.append(
            {
                "mse_loss": validation_avg_mseloss,
                "data_loss_list": validation_metrics_list,
            }
        )

        if epoch % print_every == 0:
            print(
                f"Epoch {epoch} | Train metrics loss: {avg_mse_loss = }, for 1 epoch {elapsed_time = } | Validation metrics loss: {validation_avg_mseloss = }  | Train metrics loss: {total_loss = } | Validation metrics loss: {validation_avg_loss = }"
            )

        # Early stopping check
        current_val_loss = validation_avg_loss
        if current_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(
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

    all_predictions = []
    all_targets = []
    start_time = time.time()
    with torch.no_grad():
        for datapoint in dataset:
            metabolites, targets = datapoint
            predictions = model(metabolites)

            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()

            prediction_map = {}
            target_map = {}
            for idx, reaction in enumerate(dataset.reactions):
                fluxmin = f"{reaction}_min"
                fluxmax = f"{reaction}_max"
                prediction_map[fluxmin] = predictions[idx][0]
                prediction_map[fluxmax] = predictions[idx][1]

                target_map[fluxmin] = targets[idx][0] * dataset.div_flux
                target_map[fluxmax] = targets[idx][1] * dataset.div_flux
            all_predictions.append(prediction_map)
            all_targets.append(target_map)
    end_time = time.time()

    metrics = {
        "predict_time": end_time - start_time,
        "n_samples": len(dataset),
    }
    return all_predictions, all_targets, metrics


def evaluate(
    model,
    dataset,
    device="cuda" if torch.cuda.is_available() else "cpu",
    threshold=5,
    scaler=None,
):

    all_predictions, all_targets, metrics = predict_deep(
        model=model,
        dataset=dataset,
        device=device,
    )

    if scaler is not None:
        all_predictions = scaler.inverse_transform(all_predictions)
        all_targets = scaler.inverse_transform(all_targets)
        mask = (all_predictions >= -threshold) & (
            all_predictions <= threshold
        )  # -2 ile 2 arasındaki değerleri bul
        all_predictions[mask] = 0  # Sadece bu aralıktaki değerleri 0'a çevir

    metrics["all_predictions"] = all_predictions
    metrics["all_targets"] = all_targets
    return metrics
