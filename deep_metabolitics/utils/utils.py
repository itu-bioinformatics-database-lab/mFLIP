import json
import pickle

import torch
from cobra.io import load_json_model

from deep_metabolitics.config import (
    models_dir,
    network_models_data_dir,
    outputs_dir,
    recon_path,
    synonym_mapping_path,
)

_loaded_recon3_cobra = None


def get_queries(fpath, sep=None):
    content = None
    with open(
        fpath,
        "r",
    ) as content_file:
        content = content_file.read()

    if sep is None:
        queries = [content]
    else:
        queries = content.split(sep)
    return queries


def get_pg_connection():
    import psycopg

    conn = psycopg.connect(
        host="localhost", dbname="postgres", user="baris", password="123456"
    )
    return conn


def load_recon():
    with open(recon_path) as f:
        recon_net = json.load(f)
    return recon_net


def load_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def load_pathway_metabolites_map(is_unique=False):
    fpath = network_models_data_dir / "pathway_metabolites_map_not_unique.pickle"
    pathway_metabolites_map = load_pickle(fpath=fpath)
    if is_unique:
        for pathway_name in pathway_metabolites_map.keys():
            pathway_metabolites_map[pathway_name] = list(
                set(pathway_metabolites_map[pathway_name])
            )

    return pathway_metabolites_map


def load_pathway_metabolite_abs_coefficient_map():
    fpath = network_models_data_dir / "pathway_metabolite_abs_coefficient_map.pickle"
    pathway_metabolite_abs_coefficient_map = load_pickle(fpath=fpath)
    return pathway_metabolite_abs_coefficient_map


def load_pathway_metabolite_sign_coefficient_map():
    fpath = network_models_data_dir / "pathway_metabolite_sign_coefficient_map.pickle"
    pathway_metabolite_sign_coefficient_map = load_pickle(fpath=fpath)
    return pathway_metabolite_sign_coefficient_map


def load_node2vec_embeddings(signed_weight=True):
    if signed_weight:
        fpath = outputs_dir / "node_embeddings_sign_weight.pickle"
    else:
        fpath = outputs_dir / "node_embeddings_unsign_weight.pickle"

    node2vec_embeddings = load_pickle(fpath)
    return node2vec_embeddings


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    return device


def save_network(model, fname, dir=None):
    print("SAVE NETWORK STARTING")
    if dir is None:
        dir = models_dir
    torch.save(model, dir / fname)
    print(f"SAVED MODEL {fname = }")


def load_network(fname, dir=None):
    print("SAVE NETWORK STARTING")
    if dir is None:
        dir = models_dir
    model = torch.load(dir / fname)
    print(f"Load MODEL {fname = }")
    return model


def load_pickle(fpath):
    with open(fpath, "rb") as file:
        data = pickle.load(file)
    return data


def save_pickle(data, fname, dir=None):
    if dir is None:
        dir = outputs_dir
    fpath = dir / fname
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "wb") as file:
        pickle.dump(data, file)
    print(fpath)
    return fpath


def load_cobra_network(network_model=None):
    global _loaded_recon3_cobra
    if _loaded_recon3_cobra is None:
        print("Yeni bir nesne oluşturuluyor.")
        _loaded_recon3_cobra = load_json_model(recon_path)
    else:
        print("Önceden yüklenmiş nesne döndürülüyor.")
    return _loaded_recon3_cobra


def load_metabolite_mapping(naming_file="synonym"):
    """
    Loads metabolite name mapping from different databases to recon.

    :param str naming_file: names of databases
    valid options {'kegg', 'pubChem', 'cheBl', 'hmdb', 'toy', "synonym" "new-synonym"}
    """

    with open(synonym_mapping_path) as f:
        return json.load(f)
