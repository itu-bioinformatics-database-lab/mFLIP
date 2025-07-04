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
        # norm_func=None,
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

    def __init__(self, df, ds_id, img_sz=224, file_args=None):
        from torchvision import transforms

        self.df = df
        self.ds_id = ds_id
        self.img_sz = img_sz
        self.working_dir = temp_metdit_dir / f"{img_sz}" / f"{ds_id}"

        self.file_args = self.make_file_args(
            working_dir=self.working_dir, ds_id=self.ds_id, file_args=file_args
        )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_sz, self.img_sz)
                ),  # ResNet için varsayılan girdi boyutu
                transforms.ToTensor(),
            ]
        )

        self.make_metdit_image(df=df, img_sz=self.img_sz, file_args=self.file_args)

    @staticmethod
    def make_file_args(working_dir, ds_id, file_args=None):
        if file_args is None:
            file_args = ARGS(
                # original_file_path=fpath,
                rate=0.2,  # TODO bunun neyi degistirdigine bakmak lazim, null olaylari ile ilgili bir sey
                log_func=None,
                norm_func="minmax",
                save_file_path=working_dir,
                method_type="RF",
                visualization=False,
                vis_num=None,  # 15
                # save_file_name=save_file_name,
            )

        fpath = working_dir / f"{ds_id}.csv"
        save_file_name = f"{ds_id}_after_feature_process_01"
        file_args.original_file_path = fpath
        file_args.save_file_name = save_file_name

        file_args.original_file_path = str(file_args.original_file_path)

        return file_args

    def get_image(self, df_index, index):

        fpath = self.working_dir / "data" / f"{df_index}" / f"{index}.csv"
        if not fpath.exists():
            self.make_metdit_image(
                df=self.df, img_sz=self.img_sz, file_args=self.file_args, force=True
            )
        image = np.loadtxt(fpath, delimiter=",")

        where_are_NaNs = np.isnan(image)
        image[where_are_NaNs] = 0
        # image = np.expand_dims(image, axis=0)
        image = Image.fromarray(image)
        features = self.image_transform(image)

        return features

    @staticmethod
    def make_metdit_image(df, img_sz, file_args, force=False):
        working_dir = file_args.save_file_path
        os.makedirs(working_dir, exist_ok=True)
        fpath = file_args.original_file_path
        # if not os.path.exists(fpath) or force:
        # if True:
        df.to_csv(fpath)
        file_args.original_file_path = fpath

        save_file_name = file_args.save_file_name

        # file_args = ARGS(
        #     original_file_path=str(fpath),
        #     rate=0.2,  # TODO bunun neyi degistirdigine bakmak lazim, null olaylari ile ilgili bir sey
        #     log_func=None,
        #     norm_func="minmax",
        #     save_file_path=str(working_dir),
        #     method_type="RF",
        #     visualization=False,
        #     vis_num=None,  # 15
        #     save_file_name=str(save_file_name),
        # )

        run_command(args=file_args)

        method_type = "summation"
        save_path = working_dir
        file_name = working_dir / f"post_{save_file_name}.csv"

        generate_function(method_type, img_sz, str(save_path), str(file_name))


class OwnSwipeConverter:

    def __init__(self, df, device, roll_count=1):
        self.df = df
        self.roll_count = roll_count
        self.N = self.df.shape[1]
        self.device = device

    def get_image(self, df_index, index):

        features = self.df.iloc[index].values
        matrix = np.array(
            [np.roll(features, i) for i in range(0, self.N, self.roll_count)]
        )
        matrix = torch.tensor(
            matrix, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        return matrix
