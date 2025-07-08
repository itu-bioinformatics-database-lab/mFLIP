import os
import dask.dataframe as dd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from deep_metabolitics.config import data_dir

class MultioutDataset(Dataset):
    def __init__(self, metabolite_fpath, label_fpath, batch_size=32, **kwargs):
        self.kwargs = kwargs
        
        self.df = self.load(fpath=metabolite_fpath)
        print("Metabolites are loaded.")
        self.label_df = self.load(fpath=label_fpath)
        print("Labels are loaded.")
        self.preprocess()
        self.batch_size = batch_size
        self.num_rows = len(self.df)
        self.n_metabolights = len(self.metabolite_names)
        self.n_labels = len(self.label_names)

    @staticmethod
    def load(fpath):
        if not isinstance(fpath, dd.DataFrame):
            fpath = str(fpath)
            print(f"{fpath = }")
            df = dd.read_parquet(fpath, engine="pyarrow")
        else:
            df = fpath
        return df
    
    def add_pathway_features(self):
        from deep_metabolitics.utils.utils import load_pathway_metabolites_map

        self.pathway_features = self.kwargs.get("pathway_features", False)
        if self.pathway_features:
            self.pathway_metabolites_columns = []
            pathway_metabolites_map = load_pathway_metabolites_map(is_unique=True)
            for pathway_name, metabolites in pathway_metabolites_map.items():
                intersect_metabolites = list(set(self.df.columns) & set(metabolites))
                if intersect_metabolites:
                    self.df[pathway_name + "_mean"] = self.df[intersect_metabolites].mean(axis=1)
                    self.pathway_metabolites_columns.append(pathway_name + "_mean")
    
    def _preprocess_metabolites(self, metabolites):
        self.metabolite_model = self.kwargs.get("metabolite_model")
        
        for m in metabolites:
            if m not in self.df.columns:
                self.df[m] = 0.0
        
        self.df = self.df.fillna(0.0)
        
        if self.metabolite_model is None:
            scaler = StandardScaler()
            self.metabolite_model = scaler.fit(self.df[metabolites].compute())
        
        self.df[metabolites] = self.df[metabolites].map_partitions(self.metabolite_model.transform)
        self.metabolite_names = metabolites
        self.feature_columns = metabolites

    def _preprocess_labels(self, labels):
        self.label_model = self.kwargs.get("label_model")
        
        self.label_df = self.label_df.fillna(0.0)
        
        if self.label_model is None:
            scaler = StandardScaler()
            self.label_model = scaler.fit(self.label_df[labels].compute())
        
        self.label_df[labels] = self.label_df[labels].map_partitions(self.label_model.transform)
        self.label_names = labels

    def preprocess(self):
        from deep_metabolitics.data.properties import (
            get_aycan_and_db_union_metabolites,
            get_aycan_union_metabolites,
            get_recon_metabolites,
        )
        
        self.add_pathway_features()
        
        metabolite_coverage = self.kwargs.get("metabolite_coverage", None)
        
        if metabolite_coverage == "fully":
            metabolites = get_recon_metabolites()
        elif metabolite_coverage == "aycan_union":
            metabolites = get_aycan_union_metabolites()
        elif metabolite_coverage == "aycan_union_plus_pathways":
            metabolites = get_aycan_union_metabolites()
        elif metabolite_coverage == "db_aycan_union":
            metabolites = get_aycan_and_db_union_metabolites()
        else:
            metabolites = list(self.df.columns)
        
        if self.pathway_features:
            metabolites += self.pathway_metabolites_columns
        
        metabolites = sorted(metabolites)
        label_list = sorted(self.label_df.columns)
        
        self._preprocess_metabolites(metabolites=metabolites)
        self._preprocess_labels(labels=label_list)

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_rows)
        
        batch_df = self.df.loc[start:end].compute()
        batch_labels = self.label_df.loc[start:end].compute()
        
        features = batch_df[self.feature_columns].values
        labels = batch_labels[self.label_names].values
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return features_tensor, labels_tensor
