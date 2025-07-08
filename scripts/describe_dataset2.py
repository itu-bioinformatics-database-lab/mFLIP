import os
import pandas as pd


from deep_metabolitics.config import data_dir

datasource = "pathwayfluxminmax_10_folds"
# datasource = "reactionminmax_10_folds"

input_dir = data_dir / datasource
# input_dir = "/arf/scratch/bacan/yl_tez/deep_metabolitics/data/work_workbench_metabolights_multiplied_by_factors/done/"
# foldchange_renamed_ST001000_1_0.csv
# fname_list = os.listdir(input_dir)
# fname_list = [fname for fname in fname_list if "foldchange_" in fname]
for fold in range(10):
# for fname in fname_list:
    dataset_id = f"train_{fold}"
    print(f"Processing {dataset_id}...")
    # print(f"Processing {fname}...")


    metabolomics_df = pd.read_parquet(
                input_dir
                / f"metabolomics_{dataset_id}.parquet.gzip"
            )
    metabolomics_df = metabolomics_df.T
    metabolomics_df = metabolomics_df[~(metabolomics_df > 10).any(axis=1)]
    metabolomics_df = metabolomics_df[~(metabolomics_df < -10).any(axis=1)]
    # metabolomics_df = pd.read_csv(os.path.join(input_dir, fname), index_col=0)


    print(f"{metabolomics_df.shape = }", f"{metabolomics_df.max().max() = }", f"{metabolomics_df.min().min() = }")