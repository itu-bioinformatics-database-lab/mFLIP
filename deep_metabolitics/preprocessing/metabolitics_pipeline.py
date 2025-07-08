import itertools
from collections import defaultdict

from sklearn.base import TransformerMixin
from sklearn_utils.preprocessing import FeatureMerger
from sklearn_utils.utils import average_by_label, map_dict_list

from deep_metabolitics.utils.utils import load_cobra_network as load_network_model


class ReactionDiffTransformer(TransformerMixin):
    """Scaler reaction by diff"""

    def __init__(self, network_model="recon3D", reference_label="healthy"):
        self.model = load_network_model(network_model)
        self.reference_label = reference_label

    def fit(self, X, y):
        self.healthy_flux = average_by_label(X, y, self.reference_label)
        return self

    def transform(self, X, y=None):
        transformed_data = self._transform(X=X, y=y)
        return transformed_data

    def _is_min_max(self, X):
        is_min_max = all(
            [
                self._is_valid_for_diff_with_min_max(reaction.id, x)
                for reaction in self.model.reactions
                for x in X
            ]
        )
        return is_min_max

    def _is_flux(self, X):
        is_flux = all(
            [
                self._is_valid_for_diff_with_flux(reaction.id, x)
                for reaction in self.model.reactions
                for x in X
            ]
        )
        return is_flux

    def _transform(self, X, y=None):
        return self._min_max_transform(X=X, y=y)

    def _min_max_transform(self, X, y=None):
        # TODO_BC If contition is not necessary.
        return [
            {
                reaction.id: self._reaction_flux_diff_min_max(reaction.id, x)
                for reaction in self.model.reactions
                if self._is_valid_for_diff_with_min_max(reaction.id, x)
            }
            for x in X
        ]

    def _flux_transform(self, X, y=None):
        return [
            {
                reaction.id: self._reaction_flux_diff_flux(reaction.id, x)
                for reaction in self.model.reactions
            }
            for x in X
        ]

    def _reaction_flux_diff_flux(self, r_id: str, x):
        r_name = self._r_flux(r_id)
        h = self.healthy_flux[r_name]
        r = x[r_name]

        for_score = r - h

        return for_score

    def _reaction_flux_diff_min_max(self, r_id: str, x):
        r_max, r_min = self._r_max_min(r_id)
        h_max = self.healthy_flux[r_max]
        h_min = self.healthy_flux[r_min]
        r_max, r_min = x[r_max], x[r_min]

        r_rev_max = abs(min(r_min, 0))
        r_rev_min = abs(min(r_max, 0))
        r_max = max(r_max, 0)
        r_min = max(r_min, 0)

        h_rev_max = abs(min(h_min, 0))
        h_rev_min = abs(min(h_max, 0))
        h_max = max(h_max, 0)
        h_min = max(h_min, 0)

        for_score = (r_max - h_max) + (r_min - h_min)
        rev_score = (r_rev_max - h_rev_max) + (r_rev_min - h_rev_min)

        return for_score + rev_score

    def _r_max_min(self, r_id):
        return "%s_max" % r_id, "%s_min" % r_id

    def _r_flux(self, r_id):
        return "%s_flux" % r_id

    def _is_valid_for_diff_with_min_max(self, r_id, x):
        return all(i in x and i in self.healthy_flux for i in self._r_max_min(r_id))

    def _is_valid_for_diff_with_flux(self, r_id, x):
        return self._r_flux(r_id) in self.healthy_flux


class PathwayTransformer(FeatureMerger):
    """Converts reaction level features to pathway level."""

    def __init__(self, network_model="recon3D", metrics="mean"):
        model = load_network_model(network_model)
        features = defaultdict(list)

        for r in model.reactions:
            features[r.subsystem].append(r.id)

        super().__init__(features, metrics)


class TransportPathwayElimination(TransformerMixin):

    black_list = ["Transport", "Exchange", "_"]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transformed_data = map_dict_list(
            X,
            if_func=lambda k, v: all(
                [not k.startswith(i) and k for i in self.black_list]
            ),
        )
        return transformed_data


class GNNTransformer(TransformerMixin):
    def __init__(
        self,
        model,
        graph_dataset,
        # optimizer_class=torch.optim.Adam,
        # lr=0.001,
        # weight_decay=0,
        # epochs=100,
        # batch_size=32,
    ):
        self.model = model
        self.graph_dataset = graph_dataset
        # self.optimizer_class = optimizer_class
        # self.loss_fn = loss_fn
        # self.lr = lr
        # self.weight_decay = weight_decay
        # self.epochs = epochs
        # self.batch_size = batch_size

    def fit(self, X, y):
        # # Convert data to PyTorch tensors
        # dataset = X

        # self.model, self.optimizer, self.train_metrics, _ = train(
        #     epochs=epochs,
        #     dataloader=dataloader,
        #     train_dataset=dataset,
        #     validation_dataset=dataset,
        #     model=self.model,
        #     learning_rate=learning_rate,
        #     weight_decay=weight_decay,
        #     dropout_rate=dropout_rate,
        #     n_start_layers=n_start_layers,
        #     batch_size=batch_size,
        #     logger=logger,
        #     scheduler_step_size=scheduler_step_size,
        #     scheduler_gamma=scheduler_gamma,
        #     early_stopping_patience=early_stopping_patience,
        #     early_stopping_min_delta=early_stopping_min_delta,
        #     early_stopping_metric_name=early_stopping_metric_name,
        #     fold=None,
        #     # fname=f"{experiment_fold}.pt",
        # )
        # TODO buna fit edilmis model verilmeli
        return self

    def transform(self, X, y=None):
        """
        :param X: list of dict which contains metabolic measurements.
        """
        self.model.eval()
        y_pred_list = []
        start_metabolite_index = min(
            self.graph_dataset.metabolite_name_index_map.values()
        )
        for metabolites in X:
            metabolite_features = torch.zeros(
                len(self.graph_dataset.metabolities),
                device=self.model.device,
                dtype=torch.float32,
            )
            for metabolite, metabolite_value in metabolites.items():
                metabolite_features[
                    self.graph_dataset.metabolite_name_index_map[metabolite]
                    - start_metabolite_index
                ] = metabolite_value
            metabolite_features = metabolite_features.unsqueeze(1)
            metabolite_features = metabolite_features.repeat(1, 2)
            x = torch.cat(
                [self.graph_dataset.reaction_features.clone(), metabolite_features],
                dim=0,
            )
            features = Data(
                x=x,
                edge_index=self.graph_dataset.edge_index,
                edge_weight=self.graph_dataset.edge_weights,
            )
            with torch.no_grad():
                y_pred = model(features)
                y_pred_list.append(y_pred.squeeze().cpu().tolist())
        y_pred_list = np.array(y_pred_list)

        transformed_x = []
        for fluxminmax in y_pred_list:
            row_map = {}
            for i, (rmin_value, rmax_value) in enumerate(fluxminmax):
                label_index = i * 2
                rmin_key = dataset.label_columns[label_index]
                rmax_key = dataset.label_columns[label_index + 1]
                row_map[rmax_key] = rmax_value
                row_map[rmin_key] = rmin_value
            transformed_x.append(row_map)
        return transformed_x
