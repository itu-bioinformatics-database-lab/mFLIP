import warnings

from deep_metabolitics.data.metabolight_dataset import PathwayFluxMinMaxDataset

warnings.filterwarnings("ignore")

import os
import random

import pandas as pd

from deep_metabolitics.data.properties import get_aycan_dataset_ids

from deep_metabolitics.utils.performance_metrics import PerformanceMetrics
from deep_metabolitics.config import data_dir

seed = 10
random.seed(seed)

experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")

metabolite_scaler_method = None
target_scaler_method = None
# metabolite_coverage = "fully"
metabolite_coverage = None
pathway_features = False
k_folds = 10
batch_size = 32



datasource = "pathwayfluxminmax_10_folds"

# experiment_name = f"{experiment_name}_{metabolite_scaler_method}_{target_scaler_method}_{metabolite_coverage}_{k_folds}_{batch_size}"

# ids_list = get_workbench_metabolights_dataset_ids()
# print(f"{len(ids_list) = }")


test_source_list = get_aycan_dataset_ids()

input_dir = data_dir / datasource

# for fold in list(range(k_folds)):
if True:
    fold = 0
    train_df = pd.read_parquet(
        input_dir
        / f"metabolomics_train_{fold}.parquet.gzip"
    )
    train_df = train_df.reset_index(drop=True)
    train_label_df = pd.read_parquet(
        input_dir
        / f"label_train_{fold}.parquet.gzip"
    )
    train_label_df = train_label_df.reset_index(drop=True)

    train_df = train_df.loc[~(train_df > 10).any(axis=1)]
    train_df = train_df.loc[~(train_df < -10).any(axis=1)]
    # 2. Satırlarda eksik veri oranı %90'dan küçük olanları tut
    filter_null_columns_by = 0.9
    print(f"{train_df.shape = }")
    train_df = train_df.loc[train_df.isna().mean(axis=1) < filter_null_columns_by]
    train_label_df = train_label_df.loc[train_df.index]
    print(f"{train_df.shape = }")
    print(f"{train_label_df.shape = }")

    train_prediction_min_df = pd.read_csv("/arf/scratch/bacan/yl_tez/scFEA/output/metabolomics_train_0_module98_cell115705_batch10240_LR0.008_epoch100_SCimpute_F_lambBal1_lambSca1_lambCellCor1_lambModCor_1e-2_20250504-031134_min.csv", index_col=0).fillna(0)
    train_prediction_min_df.columns = [f"{col}_min" for col in train_prediction_min_df.columns]
    train_prediction_max_df = pd.read_csv("/arf/scratch/bacan/yl_tez/scFEA/output/metabolomics_train_0_module98_cell115705_batch10240_LR0.008_epoch100_SCimpute_F_lambBal1_lambSca1_lambCellCor1_lambModCor_1e-2_20250504-031137_max.csv", index_col=0).fillna(0)
    train_prediction_max_df.columns = [f"{col}_max" for col in train_prediction_max_df.columns]
    train_prediction_df = pd.concat([train_prediction_min_df, train_prediction_max_df], axis=1)
    print(f"{train_prediction_df.shape = }")


    scaler = None
    metabolite_scaler = None
    test_all_dataset = PathwayFluxMinMaxDataset(
        dataset_ids=test_source_list,
        scaler_method=target_scaler_method,
        metabolite_scaler_method=metabolite_scaler_method,
        datasource="aycan",
        metabolite_coverage=list(train_df.columns),
        pathway_features=pathway_features,
        scaler = scaler,
        metabolite_scaler = metabolite_scaler,
        # eval_mode=False,
        run_init=True,
    )
    test_label_df = test_all_dataset.label_df
    print(f"{test_label_df.shape = }")
    test_prediction_min_df = pd.read_csv("/arf/scratch/bacan/yl_tez/scFEA/output/metabolomics_cancer_module98_cell550_batch10240_LR0.008_epoch100_SCimpute_F_lambBal1_lambSca1_lambCellCor1_lambModCor_1e-2_20250504-133307_min.csv", index_col=0).fillna(0)
    test_prediction_min_df.columns = [f"{col}_min" for col in test_prediction_min_df.columns]
    test_prediction_max_df = pd.read_csv("/arf/scratch/bacan/yl_tez/scFEA/output/metabolomics_cancer_module98_cell550_batch10240_LR0.008_epoch100_SCimpute_F_lambBal1_lambSca1_lambCellCor1_lambModCor_1e-2_20250504-150536_max.csv", index_col=0).fillna(0)
    test_prediction_max_df.columns = [f"{col}_max" for col in test_prediction_max_df.columns]
    test_prediction_df = pd.concat([test_prediction_min_df, test_prediction_max_df], axis=1)
    print(f"{test_prediction_df.shape = }")


    validation_label_df = pd.read_parquet(
        input_dir
        / f"label_test_{fold}.parquet.gzip"
    )
    print(f"{validation_label_df.shape = }")
    validation_prediction_min_df = pd.read_csv("/arf/scratch/bacan/yl_tez/scFEA/output/metabolomics_test_0_module98_cell21414_batch10240_LR0.008_epoch100_SCimpute_F_lambBal1_lambSca1_lambCellCor1_lambModCor_1e-2_20250504-232310_min.csv", index_col=0).fillna(0)
    validation_prediction_min_df.columns = [f"{col}_min" for col in validation_prediction_min_df.columns]
    validation_prediction_max_df = pd.read_csv("/arf/scratch/bacan/yl_tez/scFEA/output/metabolomics_test_0_module98_cell21414_batch10240_LR0.008_epoch100_SCimpute_F_lambBal1_lambSca1_lambCellCor1_lambModCor_1e-2_20250504-232210_max.csv", index_col=0).fillna(0)
    validation_prediction_max_df.columns = [f"{col}_max" for col in validation_prediction_max_df.columns]
    validation_prediction_df = pd.concat([validation_prediction_min_df, validation_prediction_max_df], axis=1)
    print(f"{validation_prediction_df.shape = }")


    experiment_fold = f"{experiment_name}_fold_{fold}"


    train_elapsed_time = 15202.953746318817 + 16133.848949670792
    validation_elapsed_time = 2715.0915031433105 + 2725.072834968567
    test_elapsed_time = 82.64633703231812 + 84.03532218933105


    performance_metrics = PerformanceMetrics(
        target_names=list(train_label_df.columns),
        experience_name=experiment_fold,
        train_time=train_elapsed_time,
        test_time=test_elapsed_time,
        validation_time=validation_elapsed_time,
        scaler=scaler,
    )

    true_train = train_label_df[train_label_df.columns].values
    pred_train = train_prediction_df[train_label_df.columns].values
    true_validation = validation_label_df[train_label_df.columns].values
    pred_validation = validation_prediction_df[train_label_df.columns].values
    true_test = test_label_df[train_label_df.columns].values
    pred_test = test_prediction_df[train_label_df.columns].values


    # true_train = train_label_df[train_label_df.columns].values
    performance_metrics.train_metric(y_true=true_train, y_pred=pred_train)
    performance_metrics.validation_metric(
        y_true=true_validation, y_pred=pred_validation
    )
    performance_metrics.test_metric(y_true=true_test, y_pred=pred_test)
    performance_metrics.complete()  # TODO foldlari tek dosyada tutsak guzel olur
