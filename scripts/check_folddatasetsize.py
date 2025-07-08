import os


import pandas as pd


from deep_metabolitics.config import data_dir

folddataset_dir = data_dir / "reactionminmax_10_folds"

file_name_list = os.listdir(folddataset_dir)
file_name_list = sorted(file_name_list)

for file_name in file_name_list:
    file_path = folddataset_dir / file_name
    df = pd.read_parquet(file_path)
    print(file_name, df.shape)