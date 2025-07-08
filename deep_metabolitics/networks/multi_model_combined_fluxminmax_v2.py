import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_metabolitics.config import models_dir
# from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
from deep_metabolitics.utils.utils import load_network, save_network


class MultiOutputRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiOutputRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return {"pathways_pred": x}



class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return x + self.block(x)  # Residual bağlantı: x + F(x)


class MetaboliteFCNN(nn.Module):
    def __init__(
        self,
        input_dim=5835,
        output_dim=98,
        hidden_dims=[2048, 1024, 512, 256],
        dropout_rate=0.2,
        num_residual_blocks=2,  # Her hidden layer için residual block sayısı
    ):
        super().__init__()

        # Input normalization
        # self.input_norm = nn.BatchNorm1d(input_dim)

        # İlk katman (boyut değiştiren)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        # Hidden layers with residual blocks
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Boyut değiştiren layer
            dimension_change = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
            )
            self.layers.append(dimension_change)

            # Aynı boyutta residual blocks
            residual_blocks = nn.ModuleList(
                [
                    ResidualBlock(hidden_dims[i + 1], dropout_rate)
                    for _ in range(num_residual_blocks)
                ]
            )
            self.layers.append(residual_blocks)

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # Input normalization
        # x = self.input_norm(x)

        # İlk katman
        x = self.input_layer(x)

        # Hidden layers with residual blocks
        for i in range(0, len(self.layers), 2):
            # Boyut değiştiren layer
            x = self.layers[i](x)

            # Residual blocks
            residual_blocks = self.layers[i + 1]
            for res_block in residual_blocks:
                x = res_block(x)

        # Output layer
        x = self.output(x)
        return {"pathways_pred": x}

    def loss_function(self, predictions, targets):
        """
        Kombine loss fonksiyonu
        """
        predictions = predictions["pathways_pred"]
        # Input validation
        if torch.isnan(predictions).any() or torch.isnan(targets).any():
            print("Warning: NaN values detected!")
            predictions = torch.nan_to_num(predictions, nan=0.0)
            targets = torch.nan_to_num(targets, nan=0.0)

        # L1 Loss (MAE)
        l1_loss = F.l1_loss(predictions, targets, reduction="mean")

        # MSE Loss
        mse_loss = F.mse_loss(predictions, targets, reduction="mean")

        # Huber Loss
        huber_loss = F.smooth_l1_loss(predictions, targets, reduction="mean", beta=0.1)

        # Correlation loss
        eps = 1e-8
        pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
        target_centered = targets - targets.mean(dim=0, keepdim=True)

        pred_std = torch.sqrt(torch.sum(pred_centered**2, dim=0, keepdim=True) + eps)
        target_std = torch.sqrt(
            torch.sum(target_centered**2, dim=0, keepdim=True) + eps
        )

        correlation = torch.sum(pred_centered * target_centered, dim=0) / (
            pred_std * target_std + eps
        )
        correlation_loss = 1 - correlation.mean()

        # kl_loss = self.kl_loss(predictions, targets)
        # ce_loss = F.binary_cross_entropy_with_logits(predictions, targets)

        # Total loss - weight'leri ayarla
        total_loss = (
            mse_loss  # MSE ağırlığı azaltıldı
            + l1_loss  # L1 loss eklendi
            # + ce_loss  # CE loss eklendi
            # + huber_loss  # Huber loss ağırlığı azaltıldı
            # + 1 * kl_loss  # KL loss eklendi
            # + 0.1 * correlation_loss  # Correlation loss ağırlığı azaltıldı
        )
        # total_loss = correlation_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "huber_loss": huber_loss,
            "correlation_loss": correlation_loss,
        }

    @staticmethod
    def kl_loss(predictions, targets):
        # Normalize predictions ve targets (probability distribution haline getirmek için)
        pred_dist = torch.softmax(predictions, dim=1)
        target_dist = torch.softmax(targets, dim=1)

        # KL divergence
        kl = torch.sum(target_dist * torch.log(target_dist / pred_dist), dim=1)
        return kl.mean()


class MultiTargetTrainingPipeline:
    model_out_dir = models_dir / "multi_target_training_pipeline"

    def __init__(
        self,
        reactions,
        fluxminmax_names,
        num_features=312,
        target_dim=2,
        lr=0.0001,
        weight_decay=0.01,
        tag="basic",
        dropout_rate=0.2,
        hidden_dims=[256, 64]
    ):
        self.model_out_dir.mkdir(exist_ok=True)
        
        self.num_features = num_features
        self.reactions = reactions
        self.fluxminmax_names = fluxminmax_names
        self.target_dim = target_dim
        self.lr = lr
        self.tag = tag
        
        self.working_out_dir = self.model_out_dir / self.tag
        self.working_out_dir.mkdir(exist_ok=True)
        

        # Modeller ve optimizasyon algoritmaları
        # self.models = nn.ModuleList(
        #     [
        #         MultiOutputRegressor(
        #             input_dim=num_features,
        #             output_dim=target_dim,
        #             # hidden_dims=[128, 64],
        #             # dropout_rate=0.2,
        #         )
        #         for reaction in reactions
        #     ]
        # )
        
        self.models = nn.ModuleList(
            [
                MetaboliteFCNN(
                    input_dim=num_features,
                    output_dim=target_dim,
                    hidden_dims=hidden_dims,
                    dropout_rate=dropout_rate,
                )
                for reaction in reactions
            ]
        )
        # self.models = nn.ModuleList(
        #     [MultiOutputRegressor(num_features, target_dim) for reaction in reactions]
        # )
        self.models = self.models.to(device="cuda")
        self.device = "cuda"

        self.optimizers = [
            torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
            for model in self.models
        ]
        self.loss_fn = nn.MSELoss()

    def train_single_model(self, model, optimizer, X_target, Y_target):
        """Tek bir model için eğitim"""
        optimizer.zero_grad()

        # Tahmin
        y_pred = model(X_target)  # Tek bir satır (target)
        y_pred = y_pred["pathways_pred"]

        # Kayıp hesaplama
        loss = self.loss_fn(y_pred, Y_target)

        # Geri yayılım ve optimizasyon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0)
        optimizer.step()
        

        return loss.item()

    def train_all_models(
        self, dataset, num_epochs=100, batch_size=32, num_workers=8, prefetch_factor=2, validation_dataset=None, eval_every=100
    ):
        """
        Tüm modelleri dataset kullanarak eğitir.

        Args:
            dataset: CustomDataset instance (ortak X ve reaksiyonlara özel Y içeriyor)
            num_epochs: Her model için eğitim epoch sayısı
            batch_size: Veri yükleyici batch boyutu
        """
        from torch.utils.data import DataLoader

        LOSS_TH = 0.1

        # Custom dataset'i DataLoader'a sar
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=num_workers,
            # prefetch_factor=prefetch_factor,
            # pin_memory=True
        )

        self.reaction_losses = {}
        for reaction_idx, (reaction, model, optimizer) in tqdm(
            enumerate(zip(self.reactions, self.models, self.optimizers))
        ):
            print(f"Eğitim: Model {reaction}/{len(self.reactions)}/{reaction_idx}")

            fluxmin = f"{reaction}_min"
            fluxmax = f"{reaction}_max"
            fluxmin_idx = self.fluxminmax_names.index(fluxmin)
            fluxmax_idx = self.fluxminmax_names.index(fluxmax)

            # Her reaksiyon için ayrı Y_target ve ortak X_target üzerinden eğitim yapılır
            for epoch in range(num_epochs):
                total_loss = 0.0
                for batch in dataloader:
                    X_batch, Y_batch = batch
                    # X_batch = X_batch.to(device="cuda")
                    Y_target = Y_batch[
                        :, [fluxmin_idx, fluxmax_idx]
                    ]  # Bu reaksiyona ait Y
                    # Y_target = Y_target.to(device="cuda")

                    # Eğitim adımı
                    loss = self.train_single_model(
                        model=model,
                        optimizer=optimizer,
                        X_target=X_batch,
                        Y_target=Y_target,
                    )
                    total_loss += loss

                avg_loss = total_loss / len(dataloader)
                print(
                    f"TRAIN Epoch {epoch + 1}/{num_epochs} - Model {reaction} Loss: {avg_loss:.4f}"
                )
                
                if epoch % eval_every == 0:
                    _, eval_loss = self.evaluate_single_model(reaction=reaction, dataset=validation_dataset, batch_size=len(validation_dataset), model=model, store=False)

                if eval_loss < LOSS_TH:
                    print(f"Early stopping in {epoch = }")
                    break
            save_network(
                model=model,
                fname=f"{reaction}_{self.tag}.model",
                dir=self.working_out_dir,
            )
            self.reaction_losses[reaction] = avg_loss
            print(f"TRAIN Son Kayıp (Model {reaction}): {avg_loss:.4f}")

    def compute_combined_loss(self, X, Y):
        """Tüm çıktılar üzerinde genel kayıp hesaplama"""
        # Tüm modellerin çıktıları
        all_outputs = torch.stack(
            [
                model(X[target_idx].unsqueeze(0)).squeeze(0)
                for target_idx, model in enumerate(self.models)
            ]
        )

        # Genel kayıp fonksiyonu
        mse_loss = torch.mean((all_outputs - Y) ** 2)
        regularization = torch.sum(all_outputs**2) * 1e-4  # Örnek regularization
        combined_loss = mse_loss + regularization
        return combined_loss.item()

    def save_models(self, directory="models"):
        """Modelleri kaydet"""
        import os

        os.makedirs(directory, exist_ok=True)
        for idx, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{directory}/model_{idx}.pt")
        print("Modeller kaydedildi.")

    def load_models(self, directory="models"):
        """Modelleri yükle"""
        for idx, model in enumerate(self.models):
            model.load_state_dict(torch.load(f"{directory}/model_{idx}.pt"))
        print("Modeller yüklendi.")

    def evaluate_single_model(self, reaction, dataset, batch_size, model=None, store=True):
        """
        Tek bir modeli verilen dataset ile değerlendirir.

        Args:
            reaction: Reaksiyon adı
            model_state_dict: Modelin ağırlıkları
            dataset: Değerlendirme için kullanılan veri kümesi
            batch_size: Veri yükleyici batch boyutu
            save_dir: Çıktıların kaydedileceği dizin
        """
        fluxmin = f"{reaction}_min"
        fluxmax = f"{reaction}_max"
        fluxmin_idx = self.fluxminmax_names.index(fluxmin)
        fluxmax_idx = self.fluxminmax_names.index(fluxmax)

        
        if model is None:
            model = load_network(
                fname=f"{reaction}_{self.tag}.model", dir=self.working_out_dir
            )
        model.eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for batch in dataloader:
                X_batch, Y_batch = batch
                # X_batch = X_batch.to(device="cuda")
                Y_target = Y_batch[:, [fluxmin_idx, fluxmax_idx]]
                # Y_target = Y_target.to(device="cuda")

                # Model tahmini
                predictions = model(X_batch)
                predictions = predictions["pathways_pred"]

                # Kayıp hesabı
                loss = torch.nn.functional.mse_loss(predictions, Y_target)
                total_loss += loss.item()

                # Tahminleri ve gerçek değerleri kaydet
                all_predictions.append(predictions)
                all_ground_truths.append(Y_target)

        avg_loss = total_loss / len(dataloader)
        predictions_tensor = torch.cat(all_predictions, dim=0)
        ground_truths_tensor = torch.cat(all_ground_truths, dim=0)

        
        if store:
            save_dir = self.working_out_dir / "evaluations"
            save_dir.mkdir(exist_ok=True)
            # Sonuçları kaydet
            torch.save(predictions_tensor, save_dir / f"{reaction}_predictions.pt")
            torch.save(ground_truths_tensor, save_dir / f"{reaction}_ground_truths.pt")
            torch.save(avg_loss, save_dir / f"{reaction}_loss.pt")

        print(f"EVALUATE Son Kayıp (Model {reaction}): {avg_loss:.4f}")
        return reaction, avg_loss

    def evaluate_all_models(self, dataset, batch_size=32):
        """
        Tüm modelleri paralel olarak değerlendirir ve sonuçları kaydeder.

        Args:
            dataset: Değerlendirme için kullanılan veri kümesi
            batch_size: Veri yükleyici batch boyutu
            max_processes: Aynı anda çalışacak maksimum işlem sayısı
            save_dir: Çıktıların kaydedileceği dizin
        """

        self.reaction_eval_losses = {}
        for reaction in tqdm(self.reactions):
            reaction, avg_loss = self.evaluate_single_model(
                reaction=reaction, dataset=dataset, batch_size=batch_size
            )
            self.reaction_eval_losses[reaction] = avg_loss
            print(
                f"Model {reaction} için değerlendirme tamamlandı. Son kayıp: {avg_loss:.4f}"
            )
