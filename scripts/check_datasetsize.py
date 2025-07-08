import os
import pandas as pd


folder_name = "/arf/scratch/bacan/yl_tez/deep_metabolitics/data/work_workbench_metabolights_multiplied_by_factors/done/"


sname_list = [fname.replace(".csv", "").replace("fluxminmax_renamed_", "") for fname in os.listdir(folder_name) if ("fluxminmax_" in fname) and (fname.endswith(".csv"))]

print(f"{len(sname_list) = }")
for sname in sname_list:
    try:
        # if os.path.exists()
        pathway_renamed_df = pd.read_csv(os.path.join(folder_name, f"pathway_renamed_{sname}.csv"))
        reactiondiff_renamed_df = pd.read_csv(os.path.join(folder_name, f"reactiondiff_renamed_{sname}.csv"))
        fluxminmax_renamed_df = pd.read_csv(os.path.join(folder_name, f"fluxminmax_renamed_{sname}.csv"))
        foldchange_renamed_df = pd.read_csv(os.path.join(folder_name, f"foldchange_renamed_{sname}.csv"))
        renamed_df = pd.read_csv(os.path.join(folder_name, f"renamed_{sname}.csv"))

        n_pathway_renamed = len(pathway_renamed_df)
        n_reactiondiff_renamed = len(reactiondiff_renamed_df)
        n_fluxminmax_renamed = len(fluxminmax_renamed_df)
        n_foldchange_renamed = len(foldchange_renamed_df)
        n_renamed = len(renamed_df)

        if not ((n_pathway_renamed == n_renamed) and (n_reactiondiff_renamed == n_renamed) and (n_fluxminmax_renamed == n_renamed) and (n_foldchange_renamed == n_renamed)):
            print(f"{sname = }")
            print(f"{n_pathway_renamed = }", f"{n_reactiondiff_renamed = }", f"{n_fluxminmax_renamed = }", f"{n_foldchange_renamed = }", f"{n_renamed = }")
    except Exception as e:
        print(f"ERROR: {sname = }", f"{e = }")
