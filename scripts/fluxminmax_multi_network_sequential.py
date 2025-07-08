# %% [markdown]
# Sen poster icin RMSE, MAE gibi diger metrikler uzerinden ilerle. Aycan'dan gelen farkli kanserler uzerinden sonuclarini goster. Data'lar artinca iyilesiyor mu bunlari gosterirsin. Farkli imputation metotlari ve modeller nasil perform ediyor bunlari gosterirsin. CNN'de farkli image olusturma yontemleri nasil perform ediyor bunlari gosterirsin.  R2 metrigi icin calismaya devam edersin yine tezin icin.

# %% [markdown]
# Bence, Aycan'dan veride oncelikle cross validation'lar bir test yap. Metriklerini kaydet. Cross validation'da her dataset'in 10% ununu test kismina ayir, geri kalan'la train et.
#
# Sonra ikinci asamada db'deki data'lar icin benzer calismayi ayrica yap.
#
# Sonra ikisini birlestirip, benzer sekilde tekrar calis ve metrikler buyuk veride iyilesiyor mu bir bak. Farkli modeller her durumda nasil perform ediyor, gozlemle.

# %% [markdown]
# TARGET scale edilmeyecek

# %% [markdown]
# # 1. Dataset arttikca performans nasil degisiyor?

# %%
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import os
import random
import time

import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold

from deep_metabolitics.config import all_generated_datasets_dir, aycan_full_data_dir
from deep_metabolitics.data.properties import get_all_ds_ids
from deep_metabolitics.utils.logger import create_logger
import torch.multiprocessing as mp

# from deep_metabolitics.data.properties import get_dataset_ids




# %%
from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    try:
        mp.set_start_method("spawn", force=True)  # force=True ile mevcut yöntemi zorla değiştirir
    except RuntimeError:
        pass  # Eğer zaten ayarlanmışsa, hata yok sayılır
    
    
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # %%
    experiment_name = "fluxminmax_multi_network_sequential"

    # %%
    aycan_source_list = [
        "metabData_breast",
        "metabData_ccRCC3",
        "metabData_ccRCC4",
        "metabData_coad",
        "metabData_pdac",
        "metabData_prostat",
    ]

    # %%
    metabolite_coverage = "aycan_union"

    metabolite_scaler_method = None
    target_scaler_method = None

    # %%
    generated_ds_ids = get_all_ds_ids(folder_path=all_generated_datasets_dir)
    uniform_dataset = ReactionMinMaxDataset(
        dataset_ids=generated_ds_ids,
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource="all_generated_datasets",
        metabolite_coverage=metabolite_coverage,
        pathway_features=False,
    )

    n_features = uniform_dataset.n_metabolights
    out_features = uniform_dataset.n_labels

    # %%
    reactions = [
        reaction.replace("_min", "")
        for reaction in uniform_dataset.label_df.columns
        if reaction.endswith("_min")
    ]

    # %%
    aycans_dataset = ReactionMinMaxDataset(
        dataset_ids=aycan_source_list,
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource="aycan",
        metabolite_coverage=metabolite_coverage,
        pathway_features=False,
    )


    # %%
    from deep_metabolitics.networks.multi_model_combined_fluxminmax import (
        MultiTargetTrainingPipeline,
    )

    pipeline = MultiTargetTrainingPipeline(
        reactions=reactions,
        fluxminmax_names=list(uniform_dataset.label_df.columns),
        num_features=n_features,
        target_dim=2,
        lr=0.0001,
        weight_decay=0.01,
        tag="optimization_bigbatch_bigmodel_01_lossth_2kepochs"
    )


    # %%
    # Eğitim sürecini başlat
    pipeline.train_all_models(dataset=uniform_dataset, num_epochs=2000, batch_size=1024, num_workers=2, prefetch_factor=2)
    
    pipeline.evaluate_all_models(dataset=aycans_dataset, batch_size=len(aycans_dataset))
