import random

import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import ConcatDataset

from deep_metabolitics.data.metabolight_dataset import (
    PathwayDataset,
    PathwayFluxMinMaxDataset,
    PathwayMinMaxDataset,
    ReactionMinMaxDataset,
)
from deep_metabolitics.data.properties import get_aycan_dataset_ids

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def get_fold_dataset_aycan(
    metabolite_scaler_method="quantile",
    target_scaler_method="autoscaler",
    metabolite_coverage="aycan_union",
    aycan_source_list=None,
    k_folds=10,
    filter_ds=[],
):
    if aycan_source_list is None:
        aycan_source_list = get_aycan_dataset_ids()
    aycan_map = {}
    fold_train_datasets = dict()

    for aycan_source in aycan_source_list:
        if aycan_source not in filter_ds:
            aycan_map[aycan_source] = {}

            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                PathwayDataset.load_data_aycan_csv(dataset_ids=[aycan_source])
            )
            print(aycan_source, metabolomics_df.shape)

            # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(metabolomics_df, factors_df))

            aycan_map[aycan_source]["k_fold"] = kf
            aycan_map[aycan_source]["fold_indices"] = fold_indices

            for fold, fold_index in enumerate(fold_indices):
                aycan_map[aycan_source][fold] = {}
                train_indices = fold_index[0]
                test_indices = fold_index[1]
                if fold not in fold_train_datasets:
                    fold_train_datasets[fold] = {}
                    fold_train_datasets[fold]["train_metabolomics_df"] = []
                    fold_train_datasets[fold]["train_label_df"] = []
                    fold_train_datasets[fold]["train_factors_df"] = []
                    fold_train_datasets[fold]["train_dataset_ids_df"] = []

                fold_train_datasets[fold]["train_metabolomics_df"].append(
                    metabolomics_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_label_df"].append(
                    label_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_factors_df"].append(
                    factors_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_dataset_ids_df"].append(
                    dataset_ids_df.iloc[train_indices]
                )

                store_train_dataset = PathwayDataset(
                    metabolomics_df=metabolomics_df.iloc[train_indices],
                    label_df=label_df.iloc[train_indices],
                    factors_df=factors_df.iloc[train_indices],
                    dataset_ids_df=dataset_ids_df.iloc[train_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="aycan",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )

                test_dataset = PathwayDataset(
                    metabolomics_df=metabolomics_df.iloc[test_indices],
                    label_df=label_df.iloc[test_indices],
                    factors_df=factors_df.iloc[test_indices],
                    dataset_ids_df=dataset_ids_df.iloc[test_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="aycan",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )
                aycan_map[aycan_source][fold]["test_dataset"] = test_dataset
                aycan_map[aycan_source][fold]["train_dataset"] = store_train_dataset

    fold_test_dataset_map = {}
    fold_train_dataset_map = {}

    for fold in fold_train_datasets:
        train_dataset = PathwayDataset(
            metabolomics_df=pd.concat(
                fold_train_datasets[fold]["train_metabolomics_df"]
            ),
            label_df=pd.concat(fold_train_datasets[fold]["train_label_df"]),
            factors_df=pd.concat(fold_train_datasets[fold]["train_factors_df"]),
            dataset_ids_df=pd.concat(fold_train_datasets[fold]["train_dataset_ids_df"]),
            scaler_method=target_scaler_method,
            metabolite_scaler_method=metabolite_scaler_method,
            datasource="aycan",
            metabolite_coverage=metabolite_coverage,
            pathway_features=False,
        )
        # print(train_dataset.scaler)
        fold_train_dataset_map[fold] = train_dataset

        test_datasets = []
        for aycan_source in aycan_map:
            test_dataset = aycan_map[aycan_source][fold]["test_dataset"]
            test_dataset.scaler = train_dataset.scaler
            test_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            test_dataset.init()
            test_datasets.append(test_dataset)

            store_train_dataset = aycan_map[aycan_source][fold]["train_dataset"]
            store_train_dataset.scaler = train_dataset.scaler
            store_train_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            store_train_dataset.init()

        test_all_dataset = ConcatDataset(test_datasets)
        fold_test_dataset_map[fold] = test_all_dataset

    return fold_train_dataset_map, fold_test_dataset_map, aycan_map


def get_fold_pathwatminmaxdataset_aycan(
    metabolite_scaler_method="quantile",
    target_scaler_method="autoscaler",
    metabolite_coverage="aycan_union",
    aycan_source_list=None,
    k_folds=10,
    filter_ds=[],
):
    if aycan_source_list is None:
        aycan_source_list = get_aycan_dataset_ids()
    aycan_map = {}
    fold_train_datasets = dict()

    for aycan_source in aycan_source_list:
        if aycan_source not in filter_ds:
            aycan_map[aycan_source] = {}

            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                PathwayMinMaxDataset.load_data_aycan_csv(dataset_ids=[aycan_source])
            )
            print(aycan_source, metabolomics_df.shape)

            # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(metabolomics_df, factors_df))

            aycan_map[aycan_source]["k_fold"] = kf
            aycan_map[aycan_source]["fold_indices"] = fold_indices

            for fold, fold_index in enumerate(fold_indices):
                aycan_map[aycan_source][fold] = {}
                train_indices = fold_index[0]
                test_indices = fold_index[1]
                if fold not in fold_train_datasets:
                    fold_train_datasets[fold] = {}
                    fold_train_datasets[fold]["train_metabolomics_df"] = []
                    fold_train_datasets[fold]["train_label_df"] = []
                    fold_train_datasets[fold]["train_factors_df"] = []
                    fold_train_datasets[fold]["train_dataset_ids_df"] = []

                fold_train_datasets[fold]["train_metabolomics_df"].append(
                    metabolomics_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_label_df"].append(
                    label_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_factors_df"].append(
                    factors_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_dataset_ids_df"].append(
                    dataset_ids_df.iloc[train_indices]
                )

                store_train_dataset = PathwayMinMaxDataset(
                    metabolomics_df=metabolomics_df.iloc[train_indices],
                    label_df=label_df.iloc[train_indices],
                    factors_df=factors_df.iloc[train_indices],
                    dataset_ids_df=dataset_ids_df.iloc[train_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="aycan",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )

                test_dataset = PathwayMinMaxDataset(
                    metabolomics_df=metabolomics_df.iloc[test_indices],
                    label_df=label_df.iloc[test_indices],
                    factors_df=factors_df.iloc[test_indices],
                    dataset_ids_df=dataset_ids_df.iloc[test_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="aycan",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )
                aycan_map[aycan_source][fold]["test_dataset"] = test_dataset
                aycan_map[aycan_source][fold]["train_dataset"] = store_train_dataset

    fold_test_dataset_map = {}
    fold_train_dataset_map = {}

    for fold in fold_train_datasets:
        train_dataset = PathwayMinMaxDataset(
            metabolomics_df=pd.concat(
                fold_train_datasets[fold]["train_metabolomics_df"]
            ),
            label_df=pd.concat(fold_train_datasets[fold]["train_label_df"]),
            factors_df=pd.concat(fold_train_datasets[fold]["train_factors_df"]),
            dataset_ids_df=pd.concat(fold_train_datasets[fold]["train_dataset_ids_df"]),
            scaler_method=target_scaler_method,
            metabolite_scaler_method=metabolite_scaler_method,
            datasource="aycan",
            metabolite_coverage=metabolite_coverage,
            pathway_features=False,
        )
        # print(train_dataset.scaler)
        fold_train_dataset_map[fold] = train_dataset

        test_datasets = []
        for aycan_source in aycan_map:
            test_dataset = aycan_map[aycan_source][fold]["test_dataset"]
            test_dataset.scaler = train_dataset.scaler
            test_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            test_dataset.init()
            test_datasets.append(test_dataset)

            store_train_dataset = aycan_map[aycan_source][fold]["train_dataset"]
            store_train_dataset.scaler = train_dataset.scaler
            store_train_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            store_train_dataset.init()

        test_all_dataset = ConcatDataset(test_datasets)
        fold_test_dataset_map[fold] = test_all_dataset

    return fold_train_dataset_map, fold_test_dataset_map, aycan_map


def get_fold_reactionminmaxdataset_aycan(
    metabolite_scaler_method="quantile",
    target_scaler_method="autoscaler",
    metabolite_coverage="aycan_union",
    aycan_source_list=None,
    k_folds=10,
    filter_ds=[],
):
    if aycan_source_list is None:
        aycan_source_list = get_aycan_dataset_ids()
    aycan_map = {}
    fold_train_datasets = dict()

    for aycan_source in aycan_source_list:
        if aycan_source not in filter_ds:
            aycan_map[aycan_source] = {}

            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                ReactionMinMaxDataset.load_data_aycan_csv(dataset_ids=[aycan_source])
            )
            print(aycan_source, metabolomics_df.shape)

            # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(metabolomics_df, factors_df))

            aycan_map[aycan_source]["k_fold"] = kf
            aycan_map[aycan_source]["fold_indices"] = fold_indices

            for fold, fold_index in enumerate(fold_indices):
                aycan_map[aycan_source][fold] = {}
                train_indices = fold_index[0]
                test_indices = fold_index[1]
                if fold not in fold_train_datasets:
                    fold_train_datasets[fold] = {}
                    fold_train_datasets[fold]["train_metabolomics_df"] = []
                    fold_train_datasets[fold]["train_label_df"] = []
                    fold_train_datasets[fold]["train_factors_df"] = []
                    fold_train_datasets[fold]["train_dataset_ids_df"] = []

                fold_train_datasets[fold]["train_metabolomics_df"].append(
                    metabolomics_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_label_df"].append(
                    label_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_factors_df"].append(
                    factors_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_dataset_ids_df"].append(
                    dataset_ids_df.iloc[train_indices]
                )

                store_train_dataset = ReactionMinMaxDataset(
                    metabolomics_df=metabolomics_df.iloc[train_indices],
                    label_df=label_df.iloc[train_indices],
                    factors_df=factors_df.iloc[train_indices],
                    dataset_ids_df=dataset_ids_df.iloc[train_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="aycan",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )

                test_dataset = ReactionMinMaxDataset(
                    metabolomics_df=metabolomics_df.iloc[test_indices],
                    label_df=label_df.iloc[test_indices],
                    factors_df=factors_df.iloc[test_indices],
                    dataset_ids_df=dataset_ids_df.iloc[test_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="aycan",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )
                aycan_map[aycan_source][fold]["test_dataset"] = test_dataset
                aycan_map[aycan_source][fold]["train_dataset"] = store_train_dataset

    fold_test_dataset_map = {}
    fold_train_dataset_map = {}

    for fold in fold_train_datasets:
        train_dataset = ReactionMinMaxDataset(
            metabolomics_df=pd.concat(
                fold_train_datasets[fold]["train_metabolomics_df"]
            ),
            label_df=pd.concat(fold_train_datasets[fold]["train_label_df"]),
            factors_df=pd.concat(fold_train_datasets[fold]["train_factors_df"]),
            dataset_ids_df=pd.concat(fold_train_datasets[fold]["train_dataset_ids_df"]),
            scaler_method=target_scaler_method,
            metabolite_scaler_method=metabolite_scaler_method,
            datasource="aycan",
            metabolite_coverage=metabolite_coverage,
            pathway_features=False,
        )
        # print(train_dataset.scaler)
        fold_train_dataset_map[fold] = train_dataset

        test_datasets = []
        for aycan_source in aycan_map:
            test_dataset = aycan_map[aycan_source][fold]["test_dataset"]
            test_dataset.scaler = train_dataset.scaler
            test_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            test_dataset.init()
            test_datasets.append(test_dataset)

            store_train_dataset = aycan_map[aycan_source][fold]["train_dataset"]
            store_train_dataset.scaler = train_dataset.scaler
            store_train_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            store_train_dataset.init()

        test_all_dataset = ConcatDataset(test_datasets)
        fold_test_dataset_map[fold] = test_all_dataset

    return fold_train_dataset_map, fold_test_dataset_map, aycan_map


def get_fold_pathwayfluxminmaxdataset_aycan(
    metabolite_scaler_method="quantile",
    target_scaler_method="autoscaler",
    metabolite_coverage="aycan_union",
    aycan_source_list=None,
    k_folds=10,
    filter_ds=[],
):
    if aycan_source_list is None:
        aycan_source_list = get_aycan_dataset_ids()
    aycan_map = {}
    fold_train_datasets = dict()

    for aycan_source in aycan_source_list:
        if aycan_source not in filter_ds:
            aycan_map[aycan_source] = {}

            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                PathwayFluxMinMaxDataset.load_data_aycan_csv(dataset_ids=[aycan_source])
            )
            print(aycan_source, metabolomics_df.shape)

            # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(metabolomics_df, factors_df))

            aycan_map[aycan_source]["k_fold"] = kf
            aycan_map[aycan_source]["fold_indices"] = fold_indices

            for fold, fold_index in enumerate(fold_indices):
                aycan_map[aycan_source][fold] = {}
                train_indices = fold_index[0]
                test_indices = fold_index[1]
                if fold not in fold_train_datasets:
                    fold_train_datasets[fold] = {}
                    fold_train_datasets[fold]["train_metabolomics_df"] = []
                    fold_train_datasets[fold]["train_label_df"] = []
                    fold_train_datasets[fold]["train_factors_df"] = []
                    fold_train_datasets[fold]["train_dataset_ids_df"] = []

                fold_train_datasets[fold]["train_metabolomics_df"].append(
                    metabolomics_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_label_df"].append(
                    label_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_factors_df"].append(
                    factors_df.iloc[train_indices]
                )
                fold_train_datasets[fold]["train_dataset_ids_df"].append(
                    dataset_ids_df.iloc[train_indices]
                )

                store_train_dataset = PathwayFluxMinMaxDataset(
                    metabolomics_df=metabolomics_df.iloc[train_indices],
                    label_df=label_df.iloc[train_indices],
                    factors_df=factors_df.iloc[train_indices],
                    dataset_ids_df=dataset_ids_df.iloc[train_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="aycan",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )

                test_dataset = PathwayFluxMinMaxDataset(
                    metabolomics_df=metabolomics_df.iloc[test_indices],
                    label_df=label_df.iloc[test_indices],
                    factors_df=factors_df.iloc[test_indices],
                    dataset_ids_df=dataset_ids_df.iloc[test_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="aycan",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )
                aycan_map[aycan_source][fold]["test_dataset"] = test_dataset
                aycan_map[aycan_source][fold]["train_dataset"] = store_train_dataset

    fold_test_dataset_map = {}
    fold_train_dataset_map = {}

    for fold in fold_train_datasets:
        train_dataset = PathwayFluxMinMaxDataset(
            metabolomics_df=pd.concat(
                fold_train_datasets[fold]["train_metabolomics_df"]
            ),
            label_df=pd.concat(fold_train_datasets[fold]["train_label_df"]),
            factors_df=pd.concat(fold_train_datasets[fold]["train_factors_df"]),
            dataset_ids_df=pd.concat(fold_train_datasets[fold]["train_dataset_ids_df"]),
            scaler_method=target_scaler_method,
            metabolite_scaler_method=metabolite_scaler_method,
            datasource="aycan",
            metabolite_coverage=metabolite_coverage,
            pathway_features=False,
        )
        # print(train_dataset.scaler)
        fold_train_dataset_map[fold] = train_dataset

        test_datasets = []
        for aycan_source in aycan_map:
            test_dataset = aycan_map[aycan_source][fold]["test_dataset"]
            test_dataset.scaler = train_dataset.scaler
            test_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            test_dataset.init()
            test_datasets.append(test_dataset)

            store_train_dataset = aycan_map[aycan_source][fold]["train_dataset"]
            store_train_dataset.scaler = train_dataset.scaler
            store_train_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            store_train_dataset.init()

        test_all_dataset = ConcatDataset(test_datasets)
        fold_test_dataset_map[fold] = test_all_dataset

    return fold_train_dataset_map, fold_test_dataset_map, aycan_map
