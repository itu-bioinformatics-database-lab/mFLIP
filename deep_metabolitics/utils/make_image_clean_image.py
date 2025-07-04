import os

import numpy as np
import torch
from PIL import Image

from deep_metabolitics.config import temp_metdit_dir
from deep_metabolitics.MetDIT.TransOmics.convert_by_cols_02 import generate_function
from deep_metabolitics.MetDIT.TransOmics.feature_process_01 import run_command


class ARGS:

    def __init__(
        self,
        original_file_path="generated_demo_for_metdit.csv",
        rate=0.2,
        log_func=None,
        norm_func="minmax",
        save_file_path="metdit_output_bariscan",
        method_type="RF",
        visualization=True,
        vis_num=None,  # 15
        save_file_name="demo_bariscan",
    ):
        self.original_file_path = original_file_path
        self.rate = rate
        self.log_func = log_func
        self.norm_func = norm_func
        self.save_file_path = save_file_path
        self.method_type = method_type
        self.visualization = visualization
        self.vis_num = vis_num
        self.save_file_name = save_file_name


class MetditConverter:

    def __init__(self, img_sz):
        from torchvision import transforms

        self.img_sz = img_sz

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_sz, self.img_sz)
                ),  # ResNet için varsayılan girdi boyutu
                transforms.ToTensor(),
            ]
        )

    def get_image(self, ds_id, df_index, index):
        working_dir = temp_metdit_dir / f"{self.img_sz}" / f"{ds_id}"

        folder_path = working_dir / "data" / f"{df_index}"
        fname = os.listdir(folder_path)[0]

        fpath = folder_path / fname

        image = np.loadtxt(fpath, delimiter=",")

        # where_are_NaNs = np.isnan(image)
        # image[where_are_NaNs] = 0
        # image = np.expand_dims(image, axis=0)
        image = Image.fromarray(image)
        features = self.image_transform(image)

        return features
