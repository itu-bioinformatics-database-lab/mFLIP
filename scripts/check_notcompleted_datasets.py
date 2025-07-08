import os
import shutil
import pandas as pd


folder_name = "/arf/scratch/bacan/yl_tez/deep_metabolitics/data/work_workbench_metabolights_multiplied_by_factors/done/"


sname_list = [fname.replace(".csv", "").split("renamed_")[1] for fname in os.listdir(folder_name) if ("renamed_" in fname) and (fname.endswith(".csv"))]
sname_list = list(set(sname_list))


for sname in sname_list:
    # if os.path.exists()
    if not os.path.exists(os.path.join(folder_name, f"pathway_renamed_{sname}.csv")):
        print(f"{sname = }")
        # TODO her sey bitince calistiracagiz, move edecegiz
        src_path = f"/arf/scratch/bacan/yl_tez/deep_metabolitics/data/work_workbench_metabolights_multiplied_by_factors/done/renamed_{sname}.csv"
        dest_path = f"/arf/scratch/bacan/yl_tez/deep_metabolitics/data/work_workbench_metabolights_multiplied_by_factors/renamed_{sname}.csv"
        shutil.move(src_path, dest_path)  # kaynak.txt -> hedef.txt