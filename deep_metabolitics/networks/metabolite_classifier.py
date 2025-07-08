import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaboliteClassifier(nn.Module):
    def __init__(
        self,
        input_dim=5835,
        output_dim=98,
        hidden_dims=[2048, 1024, 512],
        dropout_rate=0.5,
        use_batch_norm=True,
        use_residual=True,
        num_classes=3,
        pathway_names=None,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        self.num_classes = num_classes

        # Input normalization with momentum
        self.input_norm = nn.BatchNorm1d(input_dim, momentum=0.1)

        # First layer
        first_block = []
        first_block.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            first_block.append(nn.BatchNorm1d(hidden_dims[0], momentum=0.1))
        first_block.extend([nn.ReLU(), nn.Dropout(dropout_rate)])
        self.layers.append(nn.Sequential(*first_block))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            block = []
            block.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                block.append(nn.BatchNorm1d(hidden_dims[i + 1], momentum=0.1))
            block.extend([nn.ReLU(), nn.Dropout(dropout_rate)])

            if use_residual and hidden_dims[i] == hidden_dims[i + 1]:
                block.append(nn.Identity())

            self.layers.append(nn.Sequential(*block))

        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1], momentum=0.1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], output_dim * num_classes),
        )

        # Pathway isimlerini kaydet
        self.pathway_names = (
            pathway_names
            if pathway_names
            else [f"Pathway_{i}" for i in range(n_pathways)]
        )

    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)

        # İlk katman
        x = self.layers[0](x)

        # Gizli katmanlar
        for i, layer in enumerate(self.layers[1:]):
            if self.use_residual and hasattr(layer[-1], "in_features"):
                identity = x
                x = layer[:-1](x)
                x = x + identity
            else:
                x = layer(x)

        # Final prediction
        x = self.final_layers(x)

        # Reshape output for multi-output classification
        x = x.view(-1, self.output_dim, self.num_classes)
        return x

    def loss_function(self, predictions, targets):
        """
        Weighted cross entropy loss with label smoothing
        """
        # Convert targets to class indices ([-1, 0, 1] -> [0, 1, 2])
        targets = (targets + self.num_classes // 2).long()

        # Reshape predictions for loss calculation
        predictions = predictions.view(-1, self.num_classes)
        targets = targets.view(-1)

        # Label smoothing factor
        smoothing = 0.1

        # Create smoothed labels
        n_classes = predictions.size(1)
        one_hot = torch.zeros_like(predictions).scatter(1, targets.unsqueeze(1), 1)
        smoothed_targets = one_hot * (1 - smoothing) + smoothing / n_classes

        # Calculate loss with label smoothing
        log_probs = F.log_softmax(predictions, dim=1)
        loss = -(smoothed_targets * log_probs).sum(dim=1).mean()

        # Calculate accuracy
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == targets).float().mean()

        return {
            "loss": loss,
            "accuracy": correct,
        }

    def train_step(self, batch, device, optimizer, scheduler=None):
        """
        Geliştirilmiş training adımı
        """
        self.train()
        self = self.to(device)

        # Batch'ten veriyi ayır ve device'a taşı
        inputs, targets, factors = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Optimize
        optimizer.zero_grad(set_to_none=True)  # Daha etkili gradient sıfırlama

        # Forward pass
        predictions = self(inputs)

        # Loss hesaplama
        loss_dict = self.loss_function(predictions, targets)

        # Backward pass
        loss_dict["loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=0.5
        )  # Daha sıkı gradient clipping

        # Optimize
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]

        return {
            "loss": loss_dict["loss"].item(),
            "accuracy": loss_dict["accuracy"].item(),
            "learning_rate": current_lr,
        }

    def evaluate(self, dataloader, device):
        """
        Tüm validation/test seti için değerlendirme
        """
        self.eval()
        # Modeli device'a taşı
        self = self.to(device)

        total_loss = 0
        total_accuracy = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                # Batch'ten veriyi ayır ve device'a taşı
                inputs, targets, factors = batch  # factors ekledik
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                predictions = self(inputs)

                # Loss ve metric hesaplama
                loss_dict = self.loss_function(predictions, targets)

                total_loss += loss_dict["loss"].item()
                total_accuracy += loss_dict["accuracy"].item()

                # Tahminleri topla
                pred_classes = (
                    torch.argmax(predictions, dim=2) - 1
                )  # [0,1,2] -> [-1,0,1]
                all_predictions.append(pred_classes.cpu())
                all_targets.append(targets.cpu())

        # Ortalama metrikler
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)

        # Tüm tahminleri ve hedefleri birleştir
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Her metabolit için ayrı accuracy hesapla
        per_metabolite_accuracy = (all_predictions == all_targets).float().mean(dim=0)

        # Her sınıf için confusion matrix hesapla
        confusion_matrices = []
        for i in range(self.output_dim):
            cm = torch.zeros(3, 3)  # 3x3 confusion matrix for each metabolite
            for t in range(3):
                for p in range(3):
                    cm[t, p] = (
                        (all_targets[:, i] == (t - 1))
                        & (all_predictions[:, i] == (p - 1))
                    ).sum()
            confusion_matrices.append(cm)

        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "per_metabolite_accuracy": per_metabolite_accuracy,
            "confusion_matrices": confusion_matrices,
            "predictions": all_predictions,
            "targets": all_targets,
        }

    @staticmethod
    def get_metrics(eval_results, pathway_names=None, num_classes=3):
        """
        Evaluation sonuçlarından detaylı metrikler hesapla (sklearn kullanarak)
        """

        labels = [i - num_classes // 2 for i in range(num_classes)]

        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        predictions = eval_results["predictions"].cpu().numpy()
        targets = eval_results["targets"].cpu().numpy()

        # Her pathway için detaylı metrikler
        pathway_metrics = {}
        n_pathways = predictions.shape[1]

        for i in range(n_pathways):
            pathway_preds = predictions[:, i]
            pathway_targets = targets[:, i]

            # Confusion matrix
            cm_norm = confusion_matrix(
                pathway_targets, pathway_preds, labels=labels, normalize="true"
            )
            cm = confusion_matrix(pathway_targets, pathway_preds, labels=labels)
            # tn, fp, fn, tp = cm.ravel()

            try:
                # Temel metrikler
                accuracy = accuracy_score(
                    pathway_targets,
                    pathway_preds,
                )
                precision = precision_score(
                    pathway_targets, pathway_preds, zero_division=0, average="mean"
                )
                recall = recall_score(
                    pathway_targets, pathway_preds, zero_division=0, average="mean"
                )
                f1 = f1_score(
                    pathway_targets, pathway_preds, zero_division=0, average="mean"
                )
                # auc = roc_auc_score(pathway_targets, pathway_probs)
                mcc = matthews_corrcoef(pathway_targets, pathway_preds)

                # Specificity (True Negative Rate)
                # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            except Exception as e:
                print(f"Warning: Error calculating metrics for pathway {i}: {str(e)}")
                accuracy = precision = recall = f1 = auc = mcc = specificity = 0

            pathway_name = pathway_names[i] if pathway_names else f"Pathway_{i}"

            pathway_metrics[pathway_name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                # "specificity": float(specificity),
                "f1": float(f1),
                # "auc": float(auc),
                "mcc": float(mcc),
                # "support": {
                #     "total_samples": len(pathway_targets),
                #     "positive_samples": int(pathway_targets.sum()),
                #     "negative_samples": int(len(pathway_targets) - pathway_targets.sum())
                # },
                "confusion_matrix": list(cm),
                "confusion_matrix_norm": list(cm_norm),
            }

        # Tüm pathway'lerin ortalama metrikleri
        avg_metrics = {
            "accuracy": np.mean([m["accuracy"] for m in pathway_metrics.values()]),
            "precision": np.mean([m["precision"] for m in pathway_metrics.values()]),
            "recall": np.mean([m["recall"] for m in pathway_metrics.values()]),
            "f1": np.mean([m["f1"] for m in pathway_metrics.values()]),
            # "auc": np.mean([m["auc"] for m in pathway_metrics.values()]),
            "mcc": np.mean([m["mcc"] for m in pathway_metrics.values()]),
        }

        # Metrikleri sırala
        sorted_pathways = {
            "by_f1": sorted(
                pathway_metrics.items(), key=lambda x: x[1]["f1"], reverse=True
            ),
            # "by_auc": sorted(
            #     pathway_metrics.items(), key=lambda x: x[1]["auc"], reverse=True
            # ),
            "by_accuracy": sorted(
                pathway_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
            ),
        }

        return {
            "pathway_metrics": pathway_metrics,
            "average_metrics": avg_metrics,
            "sorted_pathways": sorted_pathways,
        }

    @staticmethod
    def plot_confusion_matrices(
        metrics,
        pathway_names=None,
        figsize=(20, 20),
        n_cols=7,
        is_norm=False,
        num_classes=3,
    ):
        """
        3 sınıflı confusion matrix'leri çizer
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        pathway_metrics = metrics["pathway_metrics"]
        n_pathways = len(pathway_metrics)
        n_rows = (n_pathways + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        # class_labels = ["Down", "No Change", "Up"]
        class_labels = [i - num_classes // 2 for i in range(num_classes)]
        if is_norm:
            cm_key = "confusion_matrix_norm"
        else:
            cm_key = "confusion_matrix"

        for idx, (pathway_name, pathway_data) in enumerate(pathway_metrics.items()):
            cm = np.array(pathway_data[cm_key])

            # Plot
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=axes[idx],
            )

            # Macro F1 skorunu hesapla
            # macro_f1 = np.mean(
            #     [
            #         pathway_data["class_metrics"][c]["f1"]
            #         for c in ["down", "no_change", "up"]
            #     ]
            # )
            macro_f1 = pathway_data["f1"]
            acc = pathway_data["accuracy"]

            # Başlık ekle
            axes[idx].set_title(
                f"{pathway_name}\nMacro F1={macro_f1:.2f}\nAccuracy={acc:.2f}",
                fontsize=8,
            )

        # Kullanılmayan subplot'ları gizle
        for idx in range(n_pathways, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_top_pathways(metrics, top_n=10, metric="f1"):
        """
        En iyi ve en kötü pathway'leri gösteren bar plot çizer

        Args:
            metrics: get_metrics'ten dönen sonuç dictionary'si
            top_n: Gösterilecek en iyi/kötü pathway sayısı
            metric: Sıralama için kullanılacak metrik ('f1', 'auc', 'accuracy')
        """
        import matplotlib.pyplot as plt

        # Pathway'leri seçilen metriğe göre sırala
        sorted_pathways = sorted(
            metrics["pathway_metrics"].items(), key=lambda x: x[1][metric], reverse=True
        )

        # En iyi ve en kötü N pathway'i al
        top_pathways = sorted_pathways[:top_n]
        bottom_pathways = sorted_pathways[-top_n:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # En iyi pathway'ler
        names = [p[0] for p in top_pathways]
        values = [p[1][metric] for p in top_pathways]
        ax1.barh(range(len(names)), values)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names)
        ax1.set_title(f"Top {top_n} Pathways")
        ax1.set_xlabel(f"{metric.upper()} Score")

        # En kötü pathway'ler
        names = [p[0] for p in bottom_pathways]
        values = [p[1][metric] for p in bottom_pathways]
        ax2.barh(range(len(names)), values)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_title(f"Bottom {top_n} Pathways")
        ax2.set_xlabel(f"{metric.upper()} Score")

        plt.tight_layout()
        return fig

    @staticmethod
    def configure_optimizers(
        model, lr=5e-5, weight_decay=1e-5, epochs=100, steps_per_epoch=100
    ):
        """
        Geliştirilmiş optimizer ve scheduler konfigürasyonu
        """
        # Adam optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Cosine Annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # İlk restart cycle uzunluğu
            T_mult=2,  # Her restart'ta cycle uzunluğunu 2'ye katla
            eta_min=1e-7,  # Minimum learning rate
        )

        return optimizer, scheduler

    @staticmethod
    def find_lr(
        model,
        train_loader,
        optimizer,
        device,
        init_value=1e-8,
        final_value=10.0,
        beta=0.98,
    ):
        """
        Learning rate finder implementation
        """
        num = len(train_loader) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        optimizer.param_groups[0]["lr"] = lr
        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch_num += 1

            # Get the loss for this mini-batch of inputs/outputs
            optimizer.zero_grad()
            loss_dict = model.train_step(
                batch=batch, device=device, optimizer=optimizer
            )
            loss = loss_dict["loss"]

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]["lr"] = lr

        return log_lrs, losses


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, verbose=True):
    """
    Tek epoch için training
    """
    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = len(train_loader)

    # Progress bar için tqdm kullanabilirsiniz
    from tqdm import tqdm

    for batch_idx, batch in enumerate(train_loader):
        # Train step
        metrics = model.train_step(batch, device, optimizer, scheduler)

        # Metrikleri topla
        epoch_loss += metrics["loss"]
        epoch_accuracy += metrics["accuracy"]

        # Her N batch'te bir progress göster
        if (batch_idx + 1) % 10 == 0:
            current_loss = epoch_loss / (batch_idx + 1)
            current_acc = epoch_accuracy / (batch_idx + 1)
            # print(
            #     f"\rEpoch {epoch} [{batch_idx+1}/{num_batches}] "
            #     f"Loss: {current_loss:.4f} "
            #     f"Accuracy: {current_acc:.4f} "
            #     f'LR: {metrics["learning_rate"]:.6f}',
            #     end="",
            # )

    # Epoch sonunda ortalama metrikleri hesapla
    avg_loss = epoch_loss / num_batches
    avg_accuracy = epoch_accuracy / num_batches

    if verbose:
        print(
            f"\nEpoch {epoch} Summary - "
            f"Average Loss: {avg_loss:.4f} "
            f"Average Accuracy: {avg_accuracy:.4f}"
        )

    return {
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "learning_rate": metrics["learning_rate"],
    }


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    lr=1e-5,
    weight_decay=0.01,
    print_epoch=10,
):
    """
    Early stopping ve daha iyi LR scheduling ile geliştirilmiş training döngüsü
    """
    optimizer, scheduler = model.configure_optimizers(
        model,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    # Early stopping parametreleri
    best_val_loss = float("inf")
    patience = 20  # 20 epoch boyunca iyileşme olmazsa dur
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            epoch % print_epoch == 0,
        )

        # Validation
        val_metrics = model.evaluate(val_loader, device)

        # Metrikleri kaydet
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["learning_rate"].append(train_metrics["learning_rate"])

        # Learning rate güncelle
        scheduler.step()

        # Early stopping kontrolü
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            # En iyi modeli geri yükle
            model.load_state_dict(best_model_state)
            break

        # Validation sonuçlarını göster
        if epoch % print_epoch == 0:
            print(
                f'Validation - Loss: {val_metrics["loss"]:.4f} '
                f'Accuracy: {val_metrics["accuracy"]:.4f}\n'
            )

    return history


def plot_training_history(history):
    """
    Eğitim geçmişini görselleştir
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Loss plot
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy plot
    ax2.plot(history["train_accuracy"], label="Train Accuracy")
    ax2.plot(history["val_accuracy"], label="Validation Accuracy")
    ax2.set_title("Accuracy History")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_training_history_every_n_epochs(history, n=10):
    """
    Eğitim geçmişini görselleştir (her 10 epoch'ta bir)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Her 10 epoch'ta bir veri noktası alacak şekilde indeksler oluştur
    total_epochs = len(history["train_loss"])
    indices = np.arange(0, total_epochs, n)
    if total_epochs - 1 not in indices:  # Son epoch'u da ekle
        indices = np.append(indices, total_epochs - 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Loss plot
    ax1.plot(
        indices, [history["train_loss"][i] for i in indices], "o-", label="Train Loss"
    )
    ax1.plot(
        indices,
        [history["val_loss"][i] for i in indices],
        "o-",
        label="Validation Loss",
    )
    ax1.set_title(f"Loss History (every {n} epochs)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(
        indices,
        [history["train_accuracy"][i] for i in indices],
        "o-",
        label="Train Accuracy",
    )
    ax2.plot(
        indices,
        [history["val_accuracy"][i] for i in indices],
        "o-",
        label="Validation Accuracy",
    )
    ax2.set_title(f"Accuracy History (every {n} epochs)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
