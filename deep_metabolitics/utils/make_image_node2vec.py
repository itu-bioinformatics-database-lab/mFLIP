import os

import numpy as np
import torch
from PIL import Image

from deep_metabolitics.config import temp_node2vec_dir
from deep_metabolitics.data.properties import get_dataset_ids
from deep_metabolitics.utils.utils import load_node2vec_embeddings

node2vec_embeddings = load_node2vec_embeddings()


def resize_sgd(image, k=224):
    np.random.seed(0)
    A = image

    # SVD hesaplama
    U, S, VT = np.linalg.svd(A.T, full_matrices=False)

    U_reduced = U[:, :k]
    S_reduced = np.diag(S[:k])
    VT_reduced = VT[:, :k]

    # Küçültülmüş temsil
    A_reduced = np.dot(U_reduced, S_reduced)  # Küçük temsil
    return np.array([A_reduced, U_reduced, VT_reduced])


def generate_image(row):
    image = []
    for column in row.keys():
        # if "mean" not in column:
        image.append(row[column] * node2vec_embeddings[column])
    image = np.array(image)

    return image


class Node2VecConverter:

    def __init__(self, img_sz=224):

        self.img_sz = img_sz

    def get_image(self, ds_id, df_index, index):

        folder_path = (
            temp_node2vec_dir / f"{self.img_sz}" / f"{ds_id}" / "data" / f"{df_index}"
        )
        fname = os.listdir(folder_path)[0]
        fpath = folder_path / fname
        image = np.load(fpath)

        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)

        return image

    @staticmethod
    def generate_images_for_df(df, img_sz, ds_id):
        for i, df_index in enumerate(df.index):
            image = generate_image(row=df.iloc[i])
            image = resize_sgd(image, k=img_sz)
            fpath = (
                temp_node2vec_dir / f"{img_sz}" / f"{ds_id}" / "data" / f"{df_index}"
            )
            fpath.mkdir(parents=True, exist_ok=True)
            fpath = fpath / f"{i}.npy"
            np.save(fpath, image)
