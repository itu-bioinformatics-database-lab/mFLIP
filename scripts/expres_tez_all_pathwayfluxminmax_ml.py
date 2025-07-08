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

from sklearn.neural_network import MLPRegressor

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")

import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from deep_metabolitics.data.fold_dataset_aycan import (
    get_fold_pathwayfluxminmaxdataset_aycan,
)
from deep_metabolitics.utils.logger import create_logger
# from deep_metabolitics.utils.trainer_fcnn import evaluate, train, warmup_training
from deep_metabolitics.utils.trainer_fcnn import (
    evaluate,
    train,
    train_sklearn,
    warmup_training,
)
from deep_metabolitics.utils.utils import load_pickle, save_pickle

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBRFRegressor

from deep_metabolitics.networks.metabolite_fcnn import MetaboliteFCNN
from deep_metabolitics.networks.metabolite_vae import MetaboliteVAE
from deep_metabolitics.networks.metabolite_vae_with_fcnn import MetaboliteVAEWithFCNN

# %%
from deep_metabolitics.networks.multiout_regressor_net_v2 import MultioutRegressorNETV2

# %% [markdown]
# # DATASET LOAD


# own_multi_model_class_list = [
#     # MultiTaskLasso,
#     # MultiTaskElasticNet,
# ]

single_model_class_list = [
    RandomForestRegressor,
    XGBRegressor,
    # XGBRFRegressor,
    # SVR,
    # MLPRegressor,
]
#

# %%
outputs_dir = "last_std_std_reactionminmax"

# %%
metabolite_scaler_method = "std"
target_scaler_method = "std"
metabolite_coverage = "aycan_union"
k_folds = 10
fold_train_dataset_map, fold_test_dataset_map, aycan_map = (
    get_fold_pathwayfluxminmaxdataset_aycan(
        metabolite_scaler_method=metabolite_scaler_method,
        target_scaler_method=target_scaler_method,
        metabolite_coverage=metabolite_coverage,
        k_folds=k_folds,
    )
)


for model_class in single_model_class_list:
    experiment_name = f"expres_tez_machinelearning_{model_class.__name__}"
    print(experiment_name)

    batch_size = 32
    metrics_map = {}
    for fold in list(range(k_folds)):
        metrics_map[fold] = {}
        print(f"Fold {fold + 1}")
        print("-------")

        train_all_dataset = fold_train_dataset_map[fold]
        test_all_dataset = fold_test_dataset_map[fold]

        scaler = train_all_dataset.scaler

        experiment_fold = f"{experiment_name}_fold_{fold}"

        n_features = train_all_dataset.n_metabolights
        out_features = train_all_dataset.n_labels

        model = MultiOutputRegressor(model_class())

        model, train_metrics, validation_metrics = train_sklearn(
            train_dataset=train_all_dataset,
            validation_dataset=test_all_dataset,
            model=model,
            pathway_names=list(train_all_dataset.label_df.columns),
        )
        metrics_map[fold]["all_train_metrics"] = train_metrics
        metrics_map[fold]["all_val_metrics"] = validation_metrics
        test_metrics = evaluate(
            model,
            test_all_dataset,
            pathway_names=list(train_all_dataset.label_df.columns),
            scaler=scaler,
            device=None,
        )
        metrics_map[fold]["all_test_metrics"] = test_metrics

        for source in aycan_map:
            test_dataset = aycan_map[source][fold]["test_dataset"]
            test_metrics = evaluate(
                model=model,
                dataset=test_dataset,
                pathway_names=list(test_dataset.label_df.columns),
                scaler=scaler,
                device=None,
            )
            metrics_map[fold][source] = test_metrics

        test_dataset = ConcatDataset(
            [
                aycan_map["metabData_ccRCC3"][fold]["test_dataset"],
                aycan_map["metabData_ccRCC4"][fold]["test_dataset"],
            ]
        )
        test_metrics = evaluate(
            model=model,
            dataset=test_dataset,
            pathway_names=list(train_all_dataset.label_df.columns),
            scaler=scaler,
            device=None,
        )
        metrics_map[fold]["metabData_ccRCC3_ccRCC4"] = test_metrics

    fpath = save_pickle(
        metrics_map,
        f"{outputs_dir}/metrics_map_10folds_{metabolite_coverage}_{metabolite_scaler_method}_{target_scaler_method}_{experiment_name}.pickle",
    )
