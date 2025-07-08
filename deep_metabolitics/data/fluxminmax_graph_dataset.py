import gc

import cobra
import numpy as np
import torch
from torch_geometric.data import Data

from deep_metabolitics.utils.utils import get_device, load_cobra_network, load_recon

# TODO: m -> r, r -> m olsun s degerleri positive olsun
# hyper edge e birden fazla giren ve cikan olacak
# TODO burada mini-batch uygulamasi yapilmamaktadir

print("deneme")


class GraphDataset:
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

    def __init__(
        self,
        dataset,
        fluxminmax_names,
        metabolite_names,
        device,
        div_flux=1,
        div_metabolities=1,
    ):
        self.device = device
        self.fluxminmax_names = fluxminmax_names
        self.metabolite_names = metabolite_names
        self.div_flux = div_flux
        self.div_metabolities = div_metabolities

        self.dataset = dataset

        self.recon3 = load_cobra_network()
        self.S_matrix = cobra.util.create_stoichiometric_matrix(self.recon3)
        self.reactions = [rxn.id for rxn in self.recon3.reactions]
        self.metabolites = [met.id for met in self.recon3.metabolites]

        (
            self.reaction_features,
            self.reaction_id_index_map,
            self.metabolite_name_index_map,
            self.edge_index,
            self.edge_weights,
        ) = self.prepare_graph_configs()

        self.reaction_features = self.reaction_features.to(device=self.device)
        self.edge_index = self.edge_index.to(device=self.device)
        self.edge_weights = self.edge_weights.to(device=self.device)
        self.S_matrix = torch.tensor(
            self.S_matrix, device=self.device, dtype=torch.float32
        )

        del self.recon3
        gc.collect()

    def get_reaction_features(self):
        reaction_id_index_map = {}
        reaction_features = []
        index = 0
        # for reaction in recon["reactions"]:
        for idx in range(len(self.reactions)):
            # pathway = reaction["subsystem"]
            # if pathway not in GraphDataset.FILTER_SUBSYSTEMS:
            reaction_name = self.reactions[idx]
            lower_bound, upper_bound = self.recon3.reactions[idx].bounds
            reaction_id_index_map[reaction_name] = idx
            reaction_features.append([lower_bound, upper_bound])
            index += 1

        reaction_features = torch.tensor(
            reaction_features,
            dtype=torch.float,
        )
        return reaction_features / 8, reaction_id_index_map

    def get_metabolities(self, reaction_id_index_map):

        start_index = len(reaction_id_index_map)
        metabolite_name_index_map = {}
        for index, name in enumerate(self.metabolites):
            metabolite_name_index_map[name] = index + start_index
        return metabolite_name_index_map

    def get_edges(self, reaction_id_index_map, metabolite_name_index_map):
        edge_source_nodes = []
        edge_target_nodes = []
        edge_weights = []

        for idx, reaction in enumerate(self.reactions):

            for metabolite, stociometry in self.recon3.reactions[
                idx
            ].metabolites.items():
                metabolite_name = metabolite.id
                if stociometry < 0:
                    edge_source_nodes.append(metabolite_name_index_map[metabolite_name])
                    edge_target_nodes.append(reaction_id_index_map[reaction])
                else:
                    edge_source_nodes.append(reaction_id_index_map[reaction])
                    edge_target_nodes.append(metabolite_name_index_map[metabolite_name])
                edge_weights.append(abs(stociometry))

        edge_index = torch.tensor(
            [
                edge_source_nodes,  # Source nodes
                edge_target_nodes,  # Target nodes
            ],
            dtype=torch.long,
        )

        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        return edge_index, edge_weights

    def prepare_graph_configs(self):
        reaction_features, reaction_id_index_map = self.get_reaction_features()
        metabolite_name_index_map = self.get_metabolities(
            reaction_id_index_map=reaction_id_index_map
        )
        edge_index, edge_weights = self.get_edges(
            reaction_id_index_map=reaction_id_index_map,
            metabolite_name_index_map=metabolite_name_index_map,
        )
        return (
            reaction_features,
            reaction_id_index_map,
            metabolite_name_index_map,
            edge_index,
            edge_weights,
        )

    def __getitem__(self, index):
        # TODO burasi bastan yazilacak.
        metabolites_data, fluxminmax_data = self.dataset[index]
        metabolite_features = []
        for (
            metabolite
        ) in (
            self.metabolites
        ):  # Graph metabolite'lerini donuyor, buradan dataset'teki yerini bulup
            if metabolite not in self.metabolite_names:
                metabolite_features.append([torch.tensor(0), torch.tensor(0)])
            else:
                idx_metabolite = self.metabolite_names.index(metabolite)
                metabolite_features.append(
                    [metabolites_data[idx_metabolite], metabolites_data[idx_metabolite]]
                )

        metabolite_features = torch.tensor(
            metabolite_features, device=self.device, dtype=torch.float32
        )

        labels = []
        for reaction in self.reactions:
            fluxmin = f"{reaction}_min"
            fluxmax = f"{reaction}_max"
            idx_fluxmin = self.fluxminmax_names.index(fluxmin)
            idx_fluxmax = self.fluxminmax_names.index(fluxmax)
            labels.append([fluxminmax_data[idx_fluxmin], fluxminmax_data[idx_fluxmax]])

        labels = torch.tensor(labels, device=self.device, dtype=torch.float32)

        x = torch.cat(
            [
                self.reaction_features.clone() / self.div_flux,
                metabolite_features / self.div_metabolities,
            ],
            dim=0,
        )

        # .unsqueeze(0)
        features = Data(x=x, edge_index=self.edge_index, edge_weight=self.edge_weights)

        return features, labels / self.div_flux

    def __len__(self):
        return len(self.dataset)

    @property
    def n_metabolights(self):
        return self.metabolomics_df.shape[1]

    @property
    def n_labels(self):
        return self.label_df.shape[1]
