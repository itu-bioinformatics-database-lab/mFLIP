import os
from pathlib import Path
os.environ["JAVA_HOME"] = "/arf/sw/lib/java/jdk-17/"


data_dir = Path("../data/")
models_dir = Path("../models/")
logs_dir = Path("../logs/")
outputs_dir = Path("../outputs/")
temp_dir = Path("../temp2/")
temp_alihoca_dir = Path("../temp_alihoca/")
temp_node2vec_dir = Path("../temp_node2vec/")
optuna_optimizers_dir = Path("../optuna_optimizers/")
oneleaveout_results_dir = Path("../oneleaveout_results/")


all_generated_datasets_dir = data_dir / "all_generated_datasets"

masked_cvae_dir = data_dir / "masked_cvae"

# aycan files
aycan_dir = data_dir / "aycan"
aycan_full_data_dir = data_dir / "aycan_full_data"
masked_cvae_dir = data_dir / "masked_cvae"
aycan_after_name_mapp_fold_change_dir = aycan_dir / "after_name_mapp_fold_change"
work_workbench_metabolights_multiplied_by_factors_dir = (
    data_dir / "work_workbench_metabolights_multiplied_by_factors" / "done"
)
aycan_pathway_diff_scores_metabolitics_pathways_diff_scores_dir = (
    aycan_dir
    / "pathways_diff_scores"
    / "pathway_diff_scores"
    / "metabolitics_pathways_diff_scores"
)

raw_data_dir = data_dir / "raw"
network_models_data_dir = data_dir / "network_models"
recon_path = network_models_data_dir / "Recon3D.json"
synonym_mapping_path = network_models_data_dir / "synonym-mapping.json"
metabolights_human_metadata_json_path = (
    network_models_data_dir / "metabolights_human_metadata.json"
)

results_dir = data_dir / "results"
generated_dir = data_dir / "generated"
uniform_aycan_generated_dir = data_dir / "uniform_aycan_generated"
raw_metabolities_dir = raw_data_dir / "metabolities_with_label"
raw_csv_dir = raw_data_dir / "csv"
raw_csv_metabolites_dir = raw_csv_dir / "metabolites"
raw_csv_pathways_dir = raw_csv_dir / "pathways"


image_log_dir = logs_dir / "images"


temp_metdit_dir = temp_dir / "metdit"


# data_path = raw_data_dir / "my_file.csv"  # use feather files if possible!!!

# customer_db_url = "sql:///customer/db/url"
# purchases_db_url = "sql:///purchases/db/url"
globals_copy = globals().copy()
for var_name, var_value in globals_copy.items():
    # print(f"{var_name}: {var_value}, {isinstance(var_value, Path)}")
    if isinstance(var_value, Path):
        if not var_value.is_file():
            var_value.mkdir(parents=True, exist_ok=True)
