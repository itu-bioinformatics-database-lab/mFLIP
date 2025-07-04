import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")


from deep_metabolitics.config import raw_csv_metabolites_dir, raw_csv_pathways_dir
from deep_metabolitics.data.metabolight_dataset import PathwayDataset
from deep_metabolitics.data.properties import get_dataset_ids
from deep_metabolitics.utils.logger import create_logger

dataset_ids = get_dataset_ids()
for dataset_id in dataset_ids:
    print(f"{dataset_id = }")
    dataset = PathwayDataset(
        dataset_ids=dataset_id,
    )
    fpath = raw_csv_metabolites_dir / f"{dataset_id}.csv"
    dataset.metabolomics_df.to_csv(fpath)

    fpath = raw_csv_pathways_dir / f"{dataset_id}.csv"
    dataset.label_df.to_csv(fpath)
