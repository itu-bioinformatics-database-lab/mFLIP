#!/bin/bash      
#SBATCH -A mvmatk
#SBATCH -p v100q
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --gres=gpu:4


#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=canb22@itu.edu.tr
# TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
# FILE_NAME=outputs2/$1_$TIMESTAMP.txt
# echo $FILE_NAME

# Load module
# module load cuda/cuda-10.2-v100q
module load Anaconda/Anaconda3-5.3.0-python3.7
source /okyanus/progs/ANACONDA/anaconda3-5.3.0-python3.7/etc/profile.d/conda.sh
conda --version
conda activate bc_deep_metabolitics
module load cuda/cuda-10.2-v100q
# source ../venv/bin/activate
# pip list
# ls
python finetune_network_by_fold_reel_data_v2_ownswipeimage_resnet.py
# # nvidia-smi
# # pip list
# # pip3 list
# # # Activate virtual env
# # source /okyanus/users/bcan/metabolitics-dev/venv/bin/activate
# # # Run Python script
# # python scripts/oneleaveout_by_finetune_network_by_fold_reel_data_v2_number32.py
# # # python main.py >> $FILE_NAME
# # # python breast-ml-pipeline.py
# # # Deactivate the env
conda deactivate
echo "DONE!"