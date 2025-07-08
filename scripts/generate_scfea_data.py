import pandas as pd


from deep_metabolitics.config import data_dir

datasource = "pathwayfluxminmax_10_folds"

input_dir = data_dir / datasource

dataset_id = "train_0"


metabolomics_df = pd.read_parquet(
            input_dir
            / f"metabolomics_{dataset_id}.parquet.gzip"
        )

metabolomics_df.T.to_csv(
    f"/arf/scratch/bacan/yl_tez/scFEA/input/metabolomics_{dataset_id}.csv",
    index=True,
    header=True,
)