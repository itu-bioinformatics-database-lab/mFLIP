import pandas as pd


from deep_metabolitics.config import data_dir
from deep_metabolitics.data.metabolight_dataset import ReactionMinMaxDataset, PathwayFluxMinMaxDataset
from deep_metabolitics.data.properties import get_aycan_dataset_ids

datasource = "pathwayfluxminmax_10_folds"

input_dir = data_dir / datasource

dataset_id = "train_0"

test_source_list = get_aycan_dataset_ids()

train_df = pd.read_parquet(
            input_dir
            / f"metabolomics_{dataset_id}.parquet.gzip"
        )


train_all_dataset = PathwayFluxMinMaxDataset(
        dataset_ids=[f"train_0"],
        scaler_method=None,
        metabolite_scaler_method=None,
        datasource=datasource,
        metabolite_coverage=None,
        pathway_features=False,
        # eval_mode=False,
        run_init=True,
    )


for ds_name in test_source_list:
    print(f"{ds_name = }")
    test_all_dataset = PathwayFluxMinMaxDataset(
        dataset_ids=[ds_name],
        scaler_method=None,
        metabolite_scaler_method=None,
        datasource="aycan",
        metabolite_coverage=train_all_dataset.metabolites_feature_columns,
        pathway_features=False,
        scaler = train_all_dataset.scaler,
        metabolite_scaler = train_all_dataset.metabolite_scaler,
        # eval_mode=False,
        run_init=True,
    )
    metabolomics_df = test_all_dataset.metabolomics_df

    metabolomics_df.T.to_csv(
        f"/arf/scratch/bacan/yl_tez/scFEA/input/metabolomics_{ds_name}.csv",
        index=True,
        header=True,
    )