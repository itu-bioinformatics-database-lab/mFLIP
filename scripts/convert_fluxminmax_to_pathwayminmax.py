from collections import defaultdict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn_utils.preprocessing import FeatureMerger
from sklearn_utils.utils import SkUtilsIO

from deep_metabolitics.config import all_generated_datasets_dir, aycan_full_data_dir
from deep_metabolitics.data.properties import get_all_ds_ids
from deep_metabolitics.utils.utils import load_cobra_network as load_network_model

FILTER_SUBSYSTEMS = [
    "Transport, peroxisomal",
    "Transport, golgi apparatus",
    "Transport, extracellular",
    "Transport, endoplasmic reticular",
    "Transport, lysosomal",
    "Transport, mitochondrial",
    "Exchange/demand reaction",
    "Transport, nuclear",
]
recon3 = load_network_model()


class PathwayRevMinMaxTransformer(FeatureMerger):
    """Converts reaction level features to pathway level."""

    def __init__(self, network_model="recon3D", metrics="mean"):
        model = load_network_model(network_model)
        features = defaultdict(list)

        for r in model.reactions:
            if r.subsystem not in FILTER_SUBSYSTEMS:
                features[f"{r.subsystem}_r_rev_max"].append(f"{r.id}_r_rev_max")
                features[f"{r.subsystem}_r_rev_min"].append(f"{r.id}_r_rev_min")
                features[f"{r.subsystem}_r_max"].append(f"{r.id}_r_max")
                features[f"{r.subsystem}_r_min"].append(f"{r.id}_r_min")
                # features[f"{r.subsystem}_min"].append(f"{r.id}_min")
                # features[f"{r.subsystem}_max"].append(f"{r.id}_max")

        super().__init__(features, metrics)


working_folder = aycan_full_data_dir

ds_id_list = get_all_ds_ids(folder_path=working_folder)


for ds_id in ds_id_list:

    fluxminmax_df = pd.read_csv(working_folder / f"fluxminmax_{ds_id}.csv", index_col=0)
    X_breast_train = []
    fluxminmax_columns = fluxminmax_df.columns.difference(["Factors"])
    for idx, row in fluxminmax_df.iterrows():
        individual = {}
        for column in fluxminmax_columns:
            individual[column] = row[column]
        X_breast_train.append(individual)

    X_breast_train_transformed = []
    for individual in X_breast_train:
        new_instance = {}
        for reaction in recon3.reactions:
            r_min = individual[f"{reaction.id}_min"]
            r_max = individual[f"{reaction.id}_max"]
            r_rev_max = abs(min(r_min, 0))
            r_rev_min = abs(min(r_max, 0))
            r_r_max = max(r_max, 0)
            r_r_min = max(r_min, 0)

            new_instance[f"{reaction.id}_r_rev_max"] = r_rev_max
            new_instance[f"{reaction.id}_r_rev_min"] = r_rev_min
            new_instance[f"{reaction.id}_r_max"] = r_r_max
            new_instance[f"{reaction.id}_r_min"] = r_r_min

            new_instance[f"{reaction.id}_max"] = r_max
            new_instance[f"{reaction.id}_min"] = r_min
        X_breast_train_transformed.append(new_instance)

    pipeline = Pipeline(
        [
            ("pathwayminmax_transformer", PathwayRevMinMaxTransformer()),
        ]
    )

    pathwayminmax = pipeline.fit_transform(X_breast_train_transformed)
    pathwayminmax_df = pd.DataFrame(pathwayminmax)
    pathwayminmax_df.to_csv(working_folder / f"pathwayminmax_{ds_id}.csv")
