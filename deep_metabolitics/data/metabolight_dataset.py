from collections import Iterable

import pandas as pd
import torch
import torch.utils
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler,
)

from deep_metabolitics.config import (
    aycan_full_data_dir,
    generated_dir,
    raw_csv_metabolites_dir,
    raw_csv_pathways_dir,
)
from deep_metabolitics.data.properties import get_aycan_union_metabolites
from deep_metabolitics.utils.make_image_alihocanin_yontemi import AlihocaConverter

# from deep_metabolitics.data.properties import get_union_metabolites
from deep_metabolitics.utils.make_image_clean_image import MetditConverter
from deep_metabolitics.utils.make_image_node2vec import Node2VecConverter
from deep_metabolitics.utils.preprocessing import own_inverse_log_scaler, own_log_scaler
from deep_metabolitics.utils.utils import get_device


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        metabolomics_df=None,
        label_df=None,
        dataset_ids=None,
        metabolite_coverage="fully",
        device=None,
        **kwargs,
    ):
        if not isinstance(dataset_ids, Iterable) or isinstance(dataset_ids, str):
            dataset_ids = [dataset_ids]
        self.scaler = kwargs.get("scaler")
        self.scaler_by = kwargs.get("scaler_by", "all")
        self.scaler_method = kwargs.get("scaler_method")
        self.scaler_map = kwargs.get("scaler_map")
        self.centroids = kwargs.get("centroids")
        self.with_factor = kwargs.get("with_factor", False)

        if metabolomics_df is None:
            metabolomics_df, label_df, dataset_ids_df, factors_df = self.load_data(
                dataset_ids=dataset_ids,
                **kwargs,
            )
        self.label_column = kwargs.get(
            "label_column"
        )  # sadece spesifik sutunlar ile calismak istiyorsak diye var

        if device is None:
            device = get_device()

        self.factors_df = factors_df
        self.metabolomics_df = metabolomics_df
        self.metabolite_coverage = metabolite_coverage
        self.label_df = label_df
        self.dataset_ids_df = dataset_ids_df
        self.dataset_ids = dataset_ids
        self.device = device
        self.is_make_image = kwargs.get("is_make_image", False)

        self.preprocess()

        if self.is_make_image:
            self.image_converter_type = kwargs.get("image_converter_type", "metdit")
            if self.image_converter_type == "metdit":
                self.make_image_by_metdit(**kwargs)
            elif self.image_converter_type == "alihoca":
                self.make_image_by_alihoca()
            elif self.image_converter_type == "node2vec":
                self.make_image_by_node2vec()

    def load_data(self, dataset_ids, **kwargs):
        from deep_metabolitics.utils.utils import load_pathway_metabolites_map

        metabolomics_df, label_df, dataset_ids_df, factors_df = self.source_load_data(
            dataset_ids=dataset_ids, **kwargs
        )

        self.pathway_features = kwargs.get("pathway_features", True)
        if self.pathway_features:
            self.pathway_metabolites_columns = []
            pathway_metabolites_map = load_pathway_metabolites_map(is_unique=True)
            for pathway_name, metabolities in pathway_metabolites_map.items():
                intersect_metabolities = metabolomics_df.columns.intersection(
                    metabolities
                )
                metabolomics_df[f"{pathway_name}_mean"] = metabolomics_df[
                    intersect_metabolities
                ].mean(axis=1)
                self.pathway_metabolites_columns.append(f"{pathway_name}_mean")
        return metabolomics_df, label_df, dataset_ids_df, factors_df

    def make_image_by_metdit(self, **kwargs):
        self.image_file_args = kwargs.get("image_file_args")
        ds_id = "_".join([str(val) for val in self.dataset_ids])
        self.img_sz = kwargs.get("img_sz", 224)

        self.image_converter = MetditConverter(
            img_sz=self.img_sz,
        )

    def make_image_by_ownswipe(self, **kwargs):
        raise Exception("Not used")
        # self.roll_count = kwargs.get(
        #     "roll_count", 1
        # )  # TODO bunu set etmek riskli suan cunku kernel ve stide degerlerini elle hesaplayip verdim
        # self.image_converter = OwnSwipeConverter(
        #     df=self.metabolomics_df, device=self.device, roll_count=self.roll_count
        # )

    def make_image_by_alihoca(self, **kwargs):
        self.img_sz = kwargs.get("img_sz", "106_1004")
        self.image_converter = AlihocaConverter(img_sz=self.img_sz)

    def make_image_by_node2vec(self, **kwargs):
        self.img_sz = kwargs.get("img_sz", 224)
        self.image_converter = Node2VecConverter(img_sz=self.img_sz)

    @staticmethod
    def get_recon_metabolites():
        from deep_metabolitics.utils.utils import load_recon

        recon_net = load_recon()
        metabolites = [m["id"] for m in recon_net["metabolites"]]
        return metabolites

    @staticmethod
    def get_union_metabolites():
        import os

        from deep_metabolitics.config import aycan_full_data_dir
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

        metabolities = list(df["metabolite_key"].values)

        if aycan_full_data_dir.exists():
            fnames = os.listdir(aycan_full_data_dir)
            for fname in fnames:
                if "foldchangescaler" in fname:
                    fpath = aycan_full_data_dir / fname
                    c = list(pd.read_csv(fpath).columns)
                    metabolities += c
        metabolities = list(set(metabolities))
        return sorted(metabolities)

    def preprocess(self):
        if self.metabolite_coverage == "fully":
            metabolites = self.get_recon_metabolites()
        elif self.metabolite_coverage == "union":
            metabolites = self.get_union_metabolites()
        elif self.metabolite_coverage == "aycan_union":
            metabolites = get_aycan_union_metabolites()
        else:
            metabolites = self.metabolomics_df.columns
        if self.pathway_features:
            metabolites += self.pathway_metabolites_columns
        if self.with_factor:
            metabolites += ["Factors"]
            self.metabolomics_df = self.metabolomics_df.join(self.factors_df)
            self.metabolomics_df["Factors"] = (
                self.metabolomics_df["Factors"] == "healthy"
            ).astype(float)
        metabolites = sorted(metabolites)
        # self.metabolomics_df = self.metabolomics_df.fillna(0)

        column_list = sorted(self.label_df.columns)
        self.label_df = self.label_df[column_list]
        self.label_df[(self.label_df > -1) & (self.label_df < 1)] = 0

        if self.label_column is not None and isinstance(self.label_column, int):
            self.label_column = self.label_df.columns[self.label_column]

        if self.label_column is not None:
            print(f"{self.label_column = }")
            self.label_df = self.label_df[[self.label_column]]

        column_list = sorted(self.metabolomics_df.columns)
        self.metabolomics_df = self.metabolomics_df[column_list]
        # for m in metabolomics_df.columns:
        # self.metabolomics_df[m] = (
        #     self.metabolomics_df[m] - self.metabolomics_df[m].min()
        # ) / (self.metabolomics_df[m].max() - self.metabolomics_df[m].min()) + 1

        # for m in metabolites:
        #     if m not in self.metabolomics_df:
        #         self.metabolomics_df[m] = 0
        for m in metabolites:
            if m not in self.metabolomics_df:
                self.metabolomics_df[m] = None

        # self.metabolomics_df /= 10
        # self.metabolomics_df += 1.1
        # self.metabolomics_df /= 2

        self.metabolomics_df += 10.1

        self.metabolomics_df = self.metabolomics_df.fillna(0)

        self.metabolomics_df = self.metabolomics_df[metabolites]

        if self.scaler_by == "all":
            self.normalize_target()
        elif self.scaler_by == "ds":
            self.normalize_target_by_dsid()
        # return self.metabolomics_df

    def normalize_target(self):
        # Denormalize etmek
        # df_denormalized = df_normalized.copy()
        # df_denormalized[df.columns] = scaler.inverse_transform(df_normalized[df.columns])

        if self.scaler is None:
            if self.scaler_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaler_method == "std":
                self.scaler = StandardScaler()
            elif self.scaler_method == "log1p":
                self.scaler = FunctionTransformer(
                    own_log_scaler, inverse_func=own_inverse_log_scaler
                )
            elif self.scaler_method == "power":
                self.scaler = PowerTransformer(method="yeo-johnson")
            else:
                return None
            self.scaler.fit(self.label_df)
        if self.scaler is not None:
            self.normalized_label_df = self.label_df.copy()
            self.normalized_label_df[self.label_df.columns] = self.scaler.transform(
                self.label_df[self.label_df.columns]
            )

    def normalize_target_by_dsid(self):
        from scipy.spatial.distance import euclidean

        if self.scaler_map is None:
            if self.scaler_method == "minmax":
                self.scaler_class = MinMaxScaler
            elif self.scaler_method == "std":
                self.scaler_class = StandardScaler
            elif self.scaler_method == "log1p":
                raise Exception("not supported")
            elif self.scaler_method == "power":
                raise Exception("not supported")
            else:
                return None
            self.scaler_map = {}
            self.centroids = {}
            for ds_id in self.dataset_ids:
                mask = self.dataset_ids_df["dataset_id"] == ds_id
                mask_index = self.dataset_ids_df[mask].index
                temp_metabolomics_df = self.metabolomics_df.loc[mask_index]
                temp_label_df = self.label_df.loc[mask_index]
                scaler = self.scaler_class()
                scaler = scaler.fit(temp_label_df)
                self.scaler_map[ds_id] = scaler
                self.centroids[ds_id] = temp_metabolomics_df.mean()
        if len(self.scaler_map) > 0:
            self.normalized_label_df = self.label_df.copy()
            for ds_id in self.dataset_ids:
                mask = self.dataset_ids_df["dataset_id"] == ds_id
                mask_index = self.dataset_ids_df[mask].index
                temp_metabolomics_df = self.metabolomics_df.loc[mask_index]
                mean_temp_metabolomics_df = temp_metabolomics_df.mean()
                temp_label_df = self.normalized_label_df.loc[mask_index]
                dist_centroids = {
                    sim_ds_id: euclidean(mean_temp_metabolomics_df, centroid)
                    for sim_ds_id, centroid in self.centroids.items()
                }
                most_sim_ds_id = min(dist_centroids, key=dist_centroids.get)
                scaler = self.scaler_map[most_sim_ds_id]
                self.normalized_label_df.loc[mask_index, self.label_df.columns] = (
                    scaler.transform(temp_label_df[self.label_df.columns])
                )

    def __getitem__(self, index):
        if self.is_make_image:
            ds_id = self.dataset_ids_df.iloc[index].values
            df_index = self.metabolomics_df.index[index]
            if isinstance(df_index, Iterable):
                feature_list = []
                for _ds_id, _df_index, _index in zip(ds_id, df_index, index):
                    if isinstance(_ds_id, Iterable):
                        _ds_id = _ds_id[0]

                    _df_index = int(_df_index)
                    features = self.image_converter.get_image(
                        ds_id=_ds_id, df_index=_df_index, index=_index
                    )
                    feature_list.append(features)
                features = torch.cat(feature_list, dim=0)
                if len(features.shape) == 3:
                    features = features.unsqueeze(1)

            else:
                if isinstance(ds_id, Iterable):
                    ds_id = ds_id[0]

                df_index = int(df_index)
                features = self.image_converter.get_image(
                    df_index=df_index, index=index, ds_id=ds_id
                )  # TODO batch olunca ne olacak emin degilim for donmek gerekebilir
        else:
            features = self.metabolomics_df.iloc[index].values
            features = torch.tensor(features, device=self.device, dtype=torch.float32)

        if hasattr(self, "normalized_label_df"):
            label = self.normalized_label_df.iloc[index].values
        else:
            label = self.label_df.iloc[index].values
        label = torch.tensor(label, device=self.device, dtype=torch.float32)

        factor = self.factors_df.iloc[index].values == "healthy"
        factor = torch.tensor(factor, device=self.device, dtype=torch.int)

        return features, label, factor

    def __len__(self):
        return len(self.metabolomics_df)

    @property
    def n_metabolights(self):
        return self.metabolomics_df.shape[1]

    @property
    def n_labels(self):
        return self.label_df.shape[1]


class PathwayDataset(BaseDataset):

    @staticmethod
    def source_load_data(
        dataset_ids,
        **kwargs,
    ):
        datasource = kwargs.get("datasource", "pg")
        if datasource == "pg":
            metabolomics_df, label_df, dataset_ids_df = (
                PathwayDataset.load_data_from_pg(dataset_ids=dataset_ids, **kwargs)
            )
        elif datasource == "csv":
            metabolomics_df, label_df, dataset_ids_df = (
                PathwayDataset.load_data_from_csv(dataset_ids=dataset_ids, **kwargs)
            )
        elif datasource == "aycan":
            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                PathwayDataset.load_data_aycan_csv(dataset_ids=dataset_ids, **kwargs)
            )
        elif datasource == "folder_pickle":
            metabolomics_df, label_df, dataset_ids_df = (
                PathwayDataset.load_data_from_folder_pickle(
                    dataset_ids=dataset_ids, **kwargs
                )
            )
        else:
            raise Exception(f"Invalid {datasource = }")

        return metabolomics_df, label_df, dataset_ids_df, factors_df

    @staticmethod
    def load_data_from_pg(
        dataset_ids,
        **kwargs,
    ):
        from deep_metabolitics.utils.utils import get_pg_connection

        query = f"""

        select a.id, a.results_pathway, md.metabolomics_data, a.dataset_id
            -- a.*
            -- , a.end_time - a.start_time duration
            -- , extract(epoch from a.end_time - a.start_time) duration_as_seconds
            -- , d.name dataset_name
            -- , md.metabolomics_data
            -- , md.is_public
        --SELECT count(*) -- 6552 -> 2188
        from analysis a
        join datasets d on a.dataset_id = d.id 
        JOIN metabolomicsdata md ON a.metabolomics_data_id = md.id
        where d.method_id=1 -- Metabolitics data only
        and a.dataset_id in ({", ".join(str(id) for id in dataset_ids)})
        and lower(a.name) not like '%my analyze%'
        and a."label" not like '%label avg%'
        """
        conn = get_pg_connection()

        df = pd.read_sql_query(sql=query, con=conn)
        df = df.set_index("id")
        DATASET_COLUMNS = ["results_pathway", "metabolomics_data", "dataset_id"]
        dataset_df = df[DATASET_COLUMNS]

        values = dataset_df["results_pathway"].apply(lambda x: x[0]).values
        label_df = pd.DataFrame(list(values), index=dataset_df.index)

        values = dataset_df["metabolomics_data"].values
        metabolomics_df = pd.DataFrame(list(values), index=dataset_df.index)

        values = dataset_df["dataset_id"].values
        dataset_ids_df = pd.DataFrame(
            list(values), index=dataset_df.index, columns=["dataset_id"]
        )

        return (
            metabolomics_df,
            label_df,
            dataset_ids_df,
        )  # TODO digerlerine de ekleyecegiz

    @staticmethod
    def load_data_from_csv(
        dataset_ids,
        **kwargs,
    ):
        metabolomics_df_list = []
        label_df_list = []
        for dataset_id in dataset_ids:
            metabolomics_df = pd.read_csv(
                raw_csv_metabolites_dir / f"{dataset_id}.csv"
            ).set_index("id")
            label_df = pd.read_csv(
                raw_csv_pathways_dir / f"{dataset_id}.csv"
            ).set_index("id")
            metabolomics_df_list.append(metabolomics_df)
            label_df_list.append(label_df)

        metabolomics_df = pd.concat(metabolomics_df_list)
        label_df = pd.concat(label_df_list)
        return metabolomics_df, label_df

    @staticmethod
    def load_data_aycan_csv(
        dataset_ids,
        **kwargs,
    ):
        factors_df_list = []
        metabolomics_df_list = []
        label_df_list = []
        dataset_ids_df_list = []

        for dataset_id in dataset_ids:
            factors_df = pd.read_csv(aycan_full_data_dir / f"{dataset_id}.csv")[
                ["Factors"]
            ]

            metabolomics_df = pd.read_csv(
                aycan_full_data_dir / f"foldchangescaler_{dataset_id}.csv", index_col=0
            )

            label_df = pd.read_csv(
                aycan_full_data_dir / f"pathway_{dataset_id}.csv", index_col=0
            )
            values = [dataset_id] * len(metabolomics_df)
            dataset_ids_df = pd.DataFrame(
                values, index=metabolomics_df.index, columns=["dataset_id"]
            )

            factors_df_list.append(factors_df)
            dataset_ids_df_list.append(dataset_ids_df)
            metabolomics_df_list.append(metabolomics_df)
            label_df_list.append(label_df)

        factors_df = pd.concat(factors_df_list).reset_index(drop=True)
        metabolomics_df = pd.concat(metabolomics_df_list).reset_index(drop=True)
        label_df = pd.concat(label_df_list).reset_index(drop=True)
        dataset_ids_df = pd.concat(dataset_ids_df_list).reset_index(drop=True)
        return metabolomics_df, label_df, dataset_ids_df, factors_df

    @staticmethod
    def load_data_from_folder_pickle(
        dataset_ids,
        **kwargs,
    ):
        import os

        folder_name = kwargs.get(
            "data_folder_name",
            "../data/1000_row_size_cover_50_iter_fixed_200_metabolites_csv",
        )
        if dataset_ids[0] is None:
            dataset_ids = [
                int(fname.replace("pathway_", "").replace(".pickle", ""))
                for fname in os.listdir(folder_name)
                if "pathway_" in fname
            ]

        metabolomics_df_list = []
        label_df_list = []
        dataset_ids_df_list = []

        for dataset_id in dataset_ids:
            metabolomics_df = pd.read_pickle(
                os.path.join(folder_name, f"foldchangescaler_{dataset_id}.pickle")
            )
            metabolomics_df = pd.DataFrame(metabolomics_df)

            label_df = pd.read_pickle(
                os.path.join(folder_name, f"pathway_{dataset_id}.pickle")
            )
            label_df = pd.DataFrame(label_df)

            values = [dataset_id] * len(metabolomics_df)
            dataset_ids_df = pd.DataFrame(
                values, index=metabolomics_df.index, columns=["dataset_id"]
            )

            dataset_ids_df_list.append(dataset_ids_df)
            metabolomics_df_list.append(metabolomics_df)
            label_df_list.append(label_df)

        metabolomics_df = pd.concat(metabolomics_df_list).reset_index(drop=True)
        label_df = pd.concat(label_df_list).reset_index(drop=True)
        dataset_ids_df = pd.concat(dataset_ids_df_list).reset_index(drop=True)
        return metabolomics_df, label_df, dataset_ids_df

    def __init__(
        self,
        metabolomics_df=None,
        label_df=None,
        dataset_ids=None,
        metabolite_coverage="fully",
        device=None,
        **kwargs,
    ):
        super(PathwayDataset, self).__init__(
            metabolomics_df=metabolomics_df,
            label_df=label_df,
            dataset_ids=dataset_ids,
            metabolite_coverage=metabolite_coverage,
            device=device,
            **kwargs,
        )


class RescaledPathwayDataset(PathwayDataset):

    @staticmethod
    def source_load_data(
        dataset_ids,
        **kwargs,
    ):
        from deep_metabolitics.config import raw_metabolities_dir

        dir_path = raw_metabolities_dir / "results"

        metabolomics_df_list = []
        label_df_list = []
        dataset_ids_df_list = []
        for id in dataset_ids:
            df = pd.DataFrame(
                pd.read_pickle(dir_path / f"foldchangescaler_{id}.pickle")
            )
            metabolomics_df_list.append(df)

            df = pd.DataFrame(pd.read_pickle(dir_path / f"pathway_{id}.pickle"))
            label_df_list.append(df)

            df = pd.DataFrame([id] * len(df), columns=["dataset_id"])
            dataset_ids_df_list.append(df)

        metabolomics_df = pd.concat(metabolomics_df_list)
        label_df = pd.concat(label_df_list)
        dataset_ids_df = pd.concat(dataset_ids_df_list)

        # for id in dataset_ids:
        #     _, label_df, dataset_ids_df = PathwayDataset.source_load_data(
        #         dataset_ids=[id]
        #     )
        #     label_df = label_df.reset_index().drop(columns=["id"])
        #     dataset_ids_df = dataset_ids_df.reset_index().drop(columns=["id"])
        #     label_df_list.append(label_df)
        #     dataset_ids_df_list.append(dataset_ids_df)

        # # df_list = [
        # #     pd.DataFrame(pd.read_pickle(dir_path / f"pathway_{id}.pickle"))
        # #     for id in dataset_ids
        # # ]
        # label_df = pd.concat(label_df_list)
        # dataset_ids_df = pd.concat(dataset_ids_df_list)

        metabolomics_df = metabolomics_df.reset_index().drop(columns=["index"])
        label_df = label_df.reset_index().drop(columns=["index"])
        dataset_ids_df = dataset_ids_df.reset_index().drop(columns=["index"])
        return metabolomics_df, label_df, dataset_ids_df

    def __init__(
        self,
        metabolomics_df=None,
        label_df=None,
        dataset_ids=None,
        metabolite_coverage="fully",
        device=None,
        **kwargs,
    ):
        super(PathwayDataset, self).__init__(
            metabolomics_df=metabolomics_df,
            label_df=label_df,
            dataset_ids=dataset_ids,
            metabolite_coverage=metabolite_coverage,
            device=device,
            **kwargs,
        )


class GeneratedPathwayDatasetv2(PathwayDataset):

    @staticmethod
    def source_load_data(
        dataset_ids,
        **kwargs,
    ):
        file_type = kwargs.get("dir_name", "json")
        dir_name = kwargs.get(
            "dir_name", "random_row_size_cover_50_iter_fixed_200_metabolites_json"
        )
        dir_path = generated_dir / dir_name
        df_list = [pd.read_json(dir_path / f"{id}.json") for id in dataset_ids]
        metabolomics_df = pd.concat(df_list)

        df_list = [
            pd.DataFrame(pd.read_pickle(dir_path / f"pathway_{id}.pickle"))
            for id in dataset_ids
        ]
        label_df = pd.concat(df_list)

        return metabolomics_df, label_df

    def __init__(
        self,
        metabolomics_df=None,
        label_df=None,
        dataset_ids=None,
        metabolite_coverage="fully",
        device=None,
        **kwargs,
    ):
        super(GeneratedPathwayDatasetv2, self).__init__(
            metabolomics_df=metabolomics_df,
            label_df=label_df,
            dataset_ids=dataset_ids,
            metabolite_coverage=metabolite_coverage,
            device=device,
            **kwargs,
        )


class GeneratedPathwayDataset(PathwayDataset):

    @staticmethod
    def source_load_data(
        dataset_ids,
        **kwargs,
    ):
        dir_name = kwargs.get(
            "dir_name", "random_row_size_cover_50_iter_fixed_200_metabolites_json"
        )
        dir_path = generated_dir / dir_name
        df_list = [pd.read_json(dir_path / f"{id}.json") for id in dataset_ids]
        metabolomics_df = pd.concat(df_list)

        df_list = [
            pd.DataFrame(pd.read_pickle(dir_path / f"pathway_{id}.pickle"))
            for id in dataset_ids
        ]
        label_df = pd.concat(df_list)

        return metabolomics_df, label_df

    def __init__(
        self,
        metabolomics_df=None,
        label_df=None,
        dataset_ids=None,
        metabolite_coverage="fully",
        device=None,
        **kwargs,
    ):
        super(GeneratedPathwayDataset, self).__init__(
            metabolomics_df=metabolomics_df,
            label_df=label_df,
            dataset_ids=dataset_ids,
            metabolite_coverage=metabolite_coverage,
            device=device,
            **kwargs,
        )


class ReactionDataset(BaseDataset):

    @staticmethod
    def source_load_data(
        dataset_ids,
        **kwargs,
    ):
        datasource = kwargs.get("datasource", "pg")
        if datasource == "pg":
            metabolomics_df, label_df, dataset_ids_df = (
                ReactionDataset.load_data_from_pg(dataset_ids=dataset_ids, **kwargs)
            )
        elif datasource == "csv":
            metabolomics_df, label_df, dataset_ids_df = (
                ReactionDataset.load_data_from_csv(dataset_ids=dataset_ids, **kwargs)
            )
        elif datasource == "aycan":
            metabolomics_df, label_df, dataset_ids_df, factors_df = (
                ReactionDataset.load_data_aycan_csv(dataset_ids=dataset_ids, **kwargs)
            )

        else:
            raise Exception(f"Invalid {datasource = }")

        return metabolomics_df, label_df, dataset_ids_df, factors_df

    @staticmethod
    def load_data_from_pg(
        dataset_ids,
        **kwargs,
    ):
        from deep_metabolitics.utils.utils import get_pg_connection

        query = f"""

        select a.id, a.results_reaction, md.metabolomics_data, a.dataset_id
            -- a.*
            -- , a.end_time - a.start_time duration
            -- , extract(epoch from a.end_time - a.start_time) duration_as_seconds
            -- , d.name dataset_name
            -- , md.metabolomics_data
            -- , md.is_public
        --SELECT count(*) -- 6552 -> 2188
        from analysis a
        join datasets d on a.dataset_id = d.id 
        JOIN metabolomicsdata md ON a.metabolomics_data_id = md.id
        where d.method_id=1 -- Metabolitics data only
        and a.dataset_id in ({", ".join(str(id) for id in dataset_ids)})
        and lower(a.name) not like '%my analyze%'
        and a."label" not like '%label avg%'
        """
        conn = get_pg_connection()

        df = pd.read_sql_query(sql=query, con=conn)
        df = df.set_index("id")
        DATASET_COLUMNS = ["results_reaction", "metabolomics_data", "dataset_id"]
        dataset_df = df[DATASET_COLUMNS]

        values = dataset_df["results_reaction"].apply(lambda x: x[0]).values
        label_df = pd.DataFrame(list(values), index=dataset_df.index)

        values = dataset_df["metabolomics_data"].values
        metabolomics_df = pd.DataFrame(list(values), index=dataset_df.index)

        values = dataset_df["dataset_id"].values
        dataset_ids_df = pd.DataFrame(
            list(values), index=dataset_df.index, columns=["dataset_id"]
        )

        return (
            metabolomics_df,
            label_df,
            dataset_ids_df,
        )  # TODO digerlerine de ekleyecegiz

    @staticmethod
    def load_data_from_csv(
        dataset_ids,
        **kwargs,
    ):
        # metabolomics_df_list = []
        # label_df_list = []
        # for dataset_id in dataset_ids:
        #     metabolomics_df = pd.read_csv(
        #         raw_csv_metabolites_dir / f"{dataset_id}.csv"
        #     ).set_index("id")
        #     label_df = pd.read_csv(
        #         raw_csv_pathways_dir / f"{dataset_id}.csv"
        #     ).set_index("id")
        #     metabolomics_df_list.append(metabolomics_df)
        #     label_df_list.append(label_df)

        # metabolomics_df = pd.concat(metabolomics_df_list)
        # label_df = pd.concat(label_df_list)
        # return metabolomics_df, label_df
        raise Exception("Not implemented yet!")

    @staticmethod
    def load_data_aycan_csv(
        dataset_ids,
        **kwargs,
    ):
        factors_df_list = []
        metabolomics_df_list = []
        label_df_list = []
        dataset_ids_df_list = []

        for dataset_id in dataset_ids:
            factors_df = pd.read_csv(aycan_full_data_dir / f"{dataset_id}.csv")[
                ["Factors"]
            ]
            metabolomics_df = pd.read_csv(
                aycan_full_data_dir / f"foldchangescaler_{dataset_id}.csv", index_col=0
            )

            label_df = pd.read_csv(
                aycan_full_data_dir / f"reactiondiff_{dataset_id}.csv",
                index_col=0,
            )
            values = [dataset_id] * len(metabolomics_df)
            dataset_ids_df = pd.DataFrame(
                values, index=metabolomics_df.index, columns=["dataset_id"]
            )

            factors_df_list.append(factors_df)
            dataset_ids_df_list.append(dataset_ids_df)
            metabolomics_df_list.append(metabolomics_df)
            label_df_list.append(label_df)

        factors_df = pd.concat(factors_df_list).reset_index(drop=True)
        metabolomics_df = pd.concat(metabolomics_df_list).reset_index(drop=True)
        label_df = pd.concat(label_df_list).reset_index(drop=True)
        dataset_ids_df = pd.concat(dataset_ids_df_list).reset_index(drop=True)
        return metabolomics_df, label_df, dataset_ids_df, factors_df

    def __init__(
        self,
        metabolomics_df=None,
        label_df=None,
        dataset_ids=None,
        metabolite_coverage="fully",
        device=None,
        **kwargs,
    ):
        super(ReactionDataset, self).__init__(
            metabolomics_df=metabolomics_df,
            label_df=label_df,
            dataset_ids=dataset_ids,
            metabolite_coverage=metabolite_coverage,
            device=device,
            **kwargs,
        )


# class ReactionDataset(BaseDataset):

#     @staticmethod
#     def source_load_data(
#         dataset_ids,
#         **kwargs,
#     ):
#         from deep_metabolitics.utils.utils import get_pg_connection

#         query = f"""

#         select a.id, a.results_reaction, md.metabolomics_data
#             -- a.*
#             -- , a.end_time - a.start_time duration
#             -- , extract(epoch from a.end_time - a.start_time) duration_as_seconds
#             -- , d.name dataset_name
#             -- , md.metabolomics_data
#             -- , md.is_public
#         --SELECT count(*) -- 6552 -> 2188
#         from analysis a
#         join datasets d on a.dataset_id = d.id
#         JOIN metabolomicsdata md ON a.metabolomics_data_id = md.id
#         where d.method_id=1 -- Metabolitics data only
#         and a.dataset_id in ({", ".join(str(id) for id in dataset_ids)})
#         and lower(a.name) not like '%my analyze%'
#         """
#         conn = get_pg_connection()

#         df = pd.read_sql_query(sql=query, con=conn)
#         df = df.set_index("id")
#         DATASET_COLUMNS = ["results_reaction", "metabolomics_data"]
#         dataset_df = df[DATASET_COLUMNS]

#         values = dataset_df["results_reaction"].apply(lambda x: x[0]).values
#         label_df = pd.DataFrame(list(values), index=dataset_df.index)

#         values = dataset_df["metabolomics_data"].values
#         metabolomics_df = pd.DataFrame(list(values), index=dataset_df.index)
#         return metabolomics_df, label_df

#     def __init__(
#         self,
#         metabolomics_df=None,
#         label_df=None,
#         dataset_ids=None,
#         metabolite_coverage="fully",
#         device=None,
#         **kwargs,
#     ):
#         super(ReactionDataset, self).__init__(
#             metabolomics_df=metabolomics_df,
#             label_df=label_df,
#             dataset_ids=dataset_ids,
#             metabolite_coverage=metabolite_coverage,
#             device=device,
#             **kwargs,
#         )


class GeneratedReactionDataset(BaseDataset):

    @staticmethod
    def source_load_data(
        dataset_ids,
        **kwargs,
    ):
        dir_name = kwargs.get(
            "dir_name", "random_row_size_cover_50_iter_fixed_200_metabolites_json"
        )
        dir_path = generated_dir / dir_name
        df_list = [pd.read_json(dir_path / f"{id}.json") for id in dataset_ids]
        metabolomics_df = pd.concat(df_list)

        df_list = [
            pd.DataFrame(pd.read_pickle(dir_path / f"reactiondiff_{id}.pickle"))
            for id in dataset_ids
        ]
        label_df = pd.concat(df_list)

        return metabolomics_df, label_df, None

    def __init__(
        self,
        metabolomics_df=None,
        label_df=None,
        dataset_ids=None,
        metabolite_coverage="fully",
        device=None,
        **kwargs,
    ):
        super(GeneratedReactionDataset, self).__init__(
            metabolomics_df=metabolomics_df,
            label_df=label_df,
            dataset_ids=dataset_ids,
            metabolite_coverage=metabolite_coverage,
            device=device,
            **kwargs,
        )


class ReactionMinMaxDataset(BaseDataset):

    @staticmethod
    def source_load_data(
        dataset_ids,
        **kwargs,
    ):
        datasource = kwargs.get("datasource", "pg")
        if datasource == "csv":
            metabolomics_df, label_df, dataset_ids_df = (
                ReactionMinMaxDataset.load_data_from_csv(
                    dataset_ids=dataset_ids, **kwargs
                )
            )
        elif datasource == "aycan":
            metabolomics_df, label_df, dataset_ids_df = (
                ReactionMinMaxDataset.load_data_aycan_csv(
                    dataset_ids=dataset_ids, **kwargs
                )
            )
        else:
            raise Exception(f"Invalid {datasource = }")

        return metabolomics_df, label_df, dataset_ids_df

    @staticmethod
    def load_data_from_csv(
        dataset_ids,
        **kwargs,
    ):
        # metabolomics_df_list = []
        # label_df_list = []
        # for dataset_id in dataset_ids:
        #     metabolomics_df = pd.read_csv(
        #         raw_csv_metabolites_dir / f"{dataset_id}.csv"
        #     ).set_index("id")
        #     label_df = pd.read_csv(
        #         raw_csv_pathways_dir / f"{dataset_id}.csv"
        #     ).set_index("id")
        #     metabolomics_df_list.append(metabolomics_df)
        #     label_df_list.append(label_df)

        # metabolomics_df = pd.concat(metabolomics_df_list)
        # label_df = pd.concat(label_df_list)
        # return metabolomics_df, label_df
        raise Exception("Not implemented yet!")

    @staticmethod
    def load_data_aycan_csv(
        dataset_ids,
        **kwargs,
    ):
        phase_str = kwargs.get("phase_str", "0_0")
        metabolomics_df_list = []
        label_df_list = []
        dataset_ids_df_list = []

        for dataset_id in dataset_ids:
            metabolomics_df = pd.read_csv(
                aycan_full_data_dir / f"foldchangescaler_{phase_str}_{dataset_id}.csv",
                index_col=0,
            )

            label_df = pd.read_csv(
                aycan_full_data_dir / f"fluxminmax_{phase_str}_{dataset_id}.csv",
                index_col=0,
            )
            values = [dataset_id] * len(metabolomics_df)
            dataset_ids_df = pd.DataFrame(
                values, index=metabolomics_df.index, columns=["dataset_id"]
            )

            dataset_ids_df_list.append(dataset_ids_df)
            metabolomics_df_list.append(metabolomics_df)
            label_df_list.append(label_df)

        metabolomics_df = pd.concat(metabolomics_df_list).reset_index(drop=True)
        label_df = pd.concat(label_df_list).reset_index(drop=True)
        dataset_ids_df = pd.concat(dataset_ids_df_list).reset_index(drop=True)
        return metabolomics_df, label_df, dataset_ids_df

    def __init__(
        self,
        metabolomics_df=None,
        label_df=None,
        dataset_ids=None,
        metabolite_coverage="fully",
        device=None,
        **kwargs,
    ):
        super(ReactionMinMaxDataset, self).__init__(
            metabolomics_df=metabolomics_df,
            label_df=label_df,
            dataset_ids=dataset_ids,
            metabolite_coverage=metabolite_coverage,
            device=device,
            **kwargs,
        )
