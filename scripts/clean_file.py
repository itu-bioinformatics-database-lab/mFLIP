import os
import pandas as pd



from deep_metabolitics.config import all_generated_datasets_dir, aycan_full_data_dir
from deep_metabolitics.data.properties import get_all_ds_ids


working_folder = all_generated_datasets_dir

fname_list = os.listdir(working_folder)

for fname in fname_list:
    fpath = os.path.join(working_folder, fname)
    try:
        n_rows = len(pd.read_csv(fpath))
        if n_rows == 0:
            print(fpath)
            os.remove(fpath)
    except:
        print(fpath)
        os.remove(fpath)
