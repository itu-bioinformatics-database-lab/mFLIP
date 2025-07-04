import os

import numpy as np
import torch
from PIL import Image

from deep_metabolitics.config import temp_alihoca_dir
from deep_metabolitics.data.properties import get_dataset_ids

# from deep_metabolitics.utils.make_image import MetditConverter
from deep_metabolitics.utils.utils import load_pathway_metabolites_map, load_recon

pathway_metabolites_map = load_pathway_metabolites_map(is_unique=True)
dataset_ids = get_dataset_ids()


def generate_image(row):
    from deep_metabolitics.data.metabolight_dataset import PathwayDataset

    # recon_metabolites = PathwayDataset.get_union_metabolites()
    recon_metabolites = PathwayDataset.get_union_metabolites()
    print(f"{len(recon_metabolites) = }")
    image = np.full((len(pathway_metabolites_map), len(recon_metabolites)), np.nan)
    for pathway_index, (pathway_name, pathway_metabolities) in enumerate(
        pathway_metabolites_map.items()
    ):
        for pathway_metabolitie in pathway_metabolities:
            if (pathway_metabolitie in row) and (
                pathway_metabolitie in recon_metabolites
            ):
                metabolite_index = recon_metabolites.index(pathway_metabolitie)
                image[pathway_index, metabolite_index] = row[pathway_metabolitie]
    return image


class AlihocaConverter:

    def __init__(self, img_sz="106_1004"):
        from torchvision import transforms

        self.img_sz = img_sz

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (106, 1004)
                    # (self.img_sz, self.img_sz)
                ),  # ResNet için varsayılan girdi boyutu
                transforms.ToTensor(),
            ]
        )

    def get_image(self, ds_id, df_index, index):

        folder_path = (
            temp_alihoca_dir / f"{self.img_sz}" / f"{ds_id}" / "data" / f"{df_index}"
        )
        fname = os.listdir(folder_path)[0]
        fpath = folder_path / fname
        image = np.load(fpath)

        image = self.preprocess_image(image=image)
        # image = np.expand_dims(image, axis=0)
        image = Image.fromarray(image)
        features = self.image_transform(image)

        return features

    @staticmethod
    def fill_na(image):
        where_are_NaNs = np.isnan(image)
        image[where_are_NaNs] = 0
        return image

    @staticmethod
    def preprocess_image(image):
        # image += 11
        image /= 10
        image += 1.1
        image /= 2

        image = AlihocaConverter.fill_na(image=image)
        return image

    @staticmethod
    def generate_images_for_df(df, img_sz, ds_id):
        for i, df_index in enumerate(df.index):
            image = generate_image(row=df.iloc[i])
            fpath = temp_alihoca_dir / f"{img_sz}" / f"{ds_id}" / "data" / f"{df_index}"
            fpath.mkdir(parents=True, exist_ok=True)
            fpath = fpath / f"{i}.npy"
            np.save(fpath, image)
