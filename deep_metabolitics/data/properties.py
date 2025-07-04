import os

import numpy as np
import pandas as pd

from deep_metabolitics.config import generated_dir, raw_csv_metabolites_dir

GENERATED_FILE_DIR_NAME = "random_row_size_cover_50_iter_fixed_200_metabolites_json"


def get_dataset_ids():
    from deep_metabolitics.utils.utils import get_pg_connection

    query = """

    SELECT DISTINCT dataset_id
    from analysis a
    join datasets d on a.dataset_id = d.id 
    JOIN metabolomicsdata md ON a.metabolomics_data_id = md.id
    where d.method_id=1 -- Metabolitics data only
    and lower(a.name) not like '%my analyze%'
    """
    conn = get_pg_connection()

    df = pd.read_sql_query(sql=query, con=conn)
    return df["dataset_id"].values


def get_dataset_ids_from_csv(dir=None):
    if dir is None:
        dir = raw_csv_metabolites_dir

    return np.array(
        [
            int(fname.replace(".csv", ""))
            for fname in os.listdir(dir)
            if fname.endswith(".csv")
        ]
    )


def get_aycan_dataset_ids():
    aycan_source_list = [
        "metabData_breast",
        "metabData_ccRCC3",
        "metabData_ccRCC4",
        "metabData_coad",
        "metabData_pdac",
        "metabData_prostat",
    ]
    return aycan_source_list


def get_generated_dataset_ids(
    dir_name="random_row_size_cover_50_iter_fixed_200_metabolites_json",
):
    dir_path = generated_dir / dir_name
    dataset_ids = [
        fname.replace(".json", "")
        for fname in os.listdir(dir_path)
        if fname.endswith(".json")
    ]
    return np.array(dataset_ids)


def get_union_metabolites():
    from deep_metabolitics.utils.utils import get_pg_connection

    query = """
    select distinct metabolite_key
    from metabolomicsdata md, json_object_keys(metabolomics_data) AS metabolite_key
    where md.id in (
        select a.metabolomics_data_id
        from analysis a
        join datasets d on a.dataset_id = d.id 
        where d.method_id=1 -- Metabolitics data only
    )

    """
    conn = get_pg_connection()

    df = pd.read_sql_query(sql=query, con=conn)
    return sorted(list(df["metabolite_key"].values))


def get_aycan_union_metabolites():
    from deep_metabolitics.config import aycan_full_data_dir

    aycan_source_list = get_aycan_dataset_ids()
    metabolites = []
    for source in aycan_source_list:
        fpath = aycan_full_data_dir / f"foldchangescaler_{source}.csv"
        df = pd.read_csv(fpath)
        metabolites.extend(df.columns)
    return sorted(list(set(metabolites)))


def get_aycan_pathway_names():
    from deep_metabolitics.config import aycan_full_data_dir

    aycan_source_list = get_aycan_dataset_ids()
    source = aycan_source_list[0]
    fpath = aycan_full_data_dir / f"pathway_{source}.csv"
    df = pd.read_csv(fpath, index_col=0)
    pathways = list(df.columns)
    return pathways
