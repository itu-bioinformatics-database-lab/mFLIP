import random

import numpy as np
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
from deep_metabolitics.data.properties import get_workbench_metabolights_dataset_ids

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def get_fold_pathway_dataset(
    metabolite_scaler_method="quantile",
    target_scaler_method="autoscaler",
    metabolite_coverage="union",
    source_list=None,
    k_folds=10,
    filter_ds=[],
):
    if source_list is None:
        source_list = get_workbench_metabolights_dataset_ids()
    map = {}
    fold_train_datasets = dict()

    for source in source_list:
        if source not in filter_ds:
            map[source] = {}

            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                PathwayDataset.load_data_csv(dataset_ids=[source])
            )
            print(source, metabolomics_df.shape)

            # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(metabolomics_df, factors_df))

            map[source]["k_fold"] = kf
            map[source]["fold_indices"] = fold_indices

            for fold, fold_index in enumerate(fold_indices):
                map[source][fold] = {}
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
                map[source][fold]["test_dataset"] = test_dataset
                map[source][fold]["train_dataset"] = store_train_dataset

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
        for source in map:
            test_dataset = map[source][fold]["test_dataset"]
            test_dataset.scaler = train_dataset.scaler
            test_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            test_dataset.init()
            test_datasets.append(test_dataset)

            store_train_dataset = map[source][fold]["train_dataset"]
            store_train_dataset.scaler = train_dataset.scaler
            store_train_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            store_train_dataset.init()

        test_all_dataset = ConcatDataset(test_datasets)
        fold_test_dataset_map[fold] = test_all_dataset

    return fold_train_dataset_map, fold_test_dataset_map, map


def get_fold_pathwatminmaxdataset(
    metabolite_scaler_method="quantile",
    target_scaler_method="autoscaler",
    metabolite_coverage="union",
    source_list=None,
    k_folds=10,
    filter_ds=[],
):
    if source_list is None:
        source_list = get_workbench_metabolights_dataset_ids()
    map = {}
    fold_train_datasets = dict()

    for source in source_list:
        if source not in filter_ds:
            map[source] = {}

            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                PathwayMinMaxDataset.load_data_csv(dataset_ids=[source])
            )
            print(source, metabolomics_df.shape)

            # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(metabolomics_df, factors_df))

            map[source]["k_fold"] = kf
            map[source]["fold_indices"] = fold_indices

            for fold, fold_index in enumerate(fold_indices):
                map[source][fold] = {}
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
                map[source][fold]["test_dataset"] = test_dataset
                map[source][fold]["train_dataset"] = store_train_dataset

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
        for source in map:
            test_dataset = map[source][fold]["test_dataset"]
            test_dataset.scaler = train_dataset.scaler
            test_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            test_dataset.init()
            test_datasets.append(test_dataset)

            store_train_dataset = map[source][fold]["train_dataset"]
            store_train_dataset.scaler = train_dataset.scaler
            store_train_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            store_train_dataset.init()

        test_all_dataset = ConcatDataset(test_datasets)
        fold_test_dataset_map[fold] = test_all_dataset

    return fold_train_dataset_map, fold_test_dataset_map, map


def get_fold_reactionminmaxdataset(
    metabolite_scaler_method="quantile",
    target_scaler_method="autoscaler",
    metabolite_coverage="fully",
    source_list=None,
    k_folds=10,
    filter_ds=[],
    fold_idx=None,
):
    if source_list is None:
        source_list = get_workbench_metabolights_dataset_ids()
    map = {}
    fold_train_datasets = dict()

    print(f"{len(source_list) = }")

    for source in source_list:
        if source not in filter_ds:
            map[source] = {}

            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                ReactionMinMaxDataset.load_data_workbench_metabolights_csv(dataset_ids=[source])
            )
            print(source, f"{metabolomics_df.shape = }", f"{factors_df.shape = }")

            # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

            try:
                fold_indices = list(kf.split(metabolomics_df, factors_df))
            except Exception as e:
                train_idx = list(np.arange(len(metabolomics_df)))
                test_idx = []
                fold_indices = [[train_idx, test_idx]] * k_folds
                print(source, e, "All of used as train dataset")


            # if len(metabolomics_df) >= k_folds:
            #     fold_indices = list(kf.split(metabolomics_df, factors_df))
            # else:
            #     train_idx = list(np.arange(len(metabolomics_df)))
            #     test_idx = []
            #     fold_indices = [[train_idx, test_idx]] * k_folds

            map[source]["k_fold"] = kf
            map[source]["fold_indices"] = fold_indices

            if fold_idx is not None:
                fold_indices = [fold_indices[fold_idx]]
            for fold, fold_index in enumerate(fold_indices):
                map[source][fold] = {}
                train_indices = fold_index[0]
                test_indices = fold_index[1]
                if fold not in fold_train_datasets:
                    fold_train_datasets[fold] = {}
                    fold_train_datasets[fold]["train_metabolomics_df"] = []
                    fold_train_datasets[fold]["train_label_df"] = []
                    fold_train_datasets[fold]["train_factors_df"] = []
                    fold_train_datasets[fold]["train_dataset_ids_df"] = []

                fold_train_datasets[fold]["train_metabolomics_df"].append(
                    metabolomics_df.loc[train_indices]
                )
                fold_train_datasets[fold]["train_label_df"].append(
                    label_df.loc[train_indices]
                )
                fold_train_datasets[fold]["train_factors_df"].append(
                    factors_df.loc[train_indices]
                )
                fold_train_datasets[fold]["train_dataset_ids_df"].append(
                    dataset_ids_df.loc[train_indices]
                )


                test_dataset = ReactionMinMaxDataset(
                    metabolomics_df=metabolomics_df.loc[test_indices],
                    label_df=label_df.loc[test_indices],
                    factors_df=factors_df.loc[test_indices],
                    dataset_ids_df=dataset_ids_df.loc[test_indices],
                    scaler_method=target_scaler_method,
                    metabolite_scaler_method=metabolite_scaler_method,
                    datasource="workbench_metabolights",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )
                map[source][fold]["test_dataset"] = test_dataset

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
            datasource="workbench_metabolights",
            metabolite_coverage=metabolite_coverage,
            pathway_features=False,
        )
        # print(train_dataset.scaler)
        fold_train_dataset_map[fold] = train_dataset

        test_datasets = []
        for source in map:
            test_dataset = map[source][fold]["test_dataset"]
            test_dataset.scaler = train_dataset.scaler
            test_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            test_dataset.init()
            test_datasets.append(test_dataset)

        test_all_dataset = ConcatDataset(test_datasets)
        fold_test_dataset_map[fold] = test_all_dataset

    return fold_train_dataset_map, fold_test_dataset_map, map


def get_fold_pathwayfluxminmaxdataset(
    metabolite_scaler_method="quantile",
    target_scaler_method="autoscaler",
    metabolite_coverage="union",
    source_list=None,
    k_folds=10,
    filter_ds=[],
):
    if source_list is None:
        source_list = get_workbench_metabolights_dataset_ids()
    map = {}
    fold_train_datasets = dict()

    for source in source_list:
        if source not in filter_ds:
            map[source] = {}

            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                PathwayFluxMinMaxDataset.load_data_workbench_metabolights_csv(dataset_ids=[source])
            )
            print(source, metabolomics_df.shape)

            # kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

            if len(metabolomics_df) >= k_folds:
                fold_indices = list(kf.split(metabolomics_df, factors_df))
            else:
                train_idx = list(np.arange(len(metabolomics_df)))
                test_idx = []
                fold_indices = [[train_idx, test_idx]] * k_folds

            map[source]["k_fold"] = kf
            map[source]["fold_indices"] = fold_indices

            for fold, fold_index in enumerate(fold_indices):
                map[source][fold] = {}
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
                    datasource="workbench_metabolights",
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
                    datasource="workbench_metabolights",
                    metabolite_coverage=metabolite_coverage,
                    pathway_features=False,
                    eval_mode=True,
                    run_init=False,
                )
                map[source][fold]["test_dataset"] = test_dataset
                map[source][fold]["train_dataset"] = store_train_dataset

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
            datasource="workbench_metabolights",
            metabolite_coverage=metabolite_coverage,
            pathway_features=False,
        )
        # print(train_dataset.scaler)
        fold_train_dataset_map[fold] = train_dataset

        test_datasets = []
        for source in map:
            test_dataset = map[source][fold]["test_dataset"]
            test_dataset.scaler = train_dataset.scaler
            test_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            test_dataset.init()
            test_datasets.append(test_dataset)

            store_train_dataset = map[source][fold]["train_dataset"]
            store_train_dataset.scaler = train_dataset.scaler
            store_train_dataset.metabolite_scaler = train_dataset.metabolite_scaler
            store_train_dataset.init()

        test_all_dataset = ConcatDataset(test_datasets)
        fold_test_dataset_map[fold] = test_all_dataset

    return fold_train_dataset_map, fold_test_dataset_map, map
