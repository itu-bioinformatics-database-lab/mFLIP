#!/bin/bash      
#SBATCH --account=bacan
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pm_wm_pathwayfluxminmax_TabNetModelConfig_pandas
#SBATCH --partition=barbun-cuda
#SBATCH --ntasks-per-node=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=canb22@itu.edu.tr
# TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
# FILE_NAME=outputs2/$1_$TIMESTAMP.txt
# echo $FILE_NAME




# Load module

module purge
module load apps/truba-ai/gpu-2024.0
module load lib/cuda/12.4
# module load comp/python/ai-tools-kolyoz-1.0

source ../venv/bin/activate

# pip list


python pm_wm_pathwayfluxminmax_TabNetModelConfig_pandas.py
nvidia-smi
echo "pm_wm_pathwayfluxminmax_TabNetModelConfig_pandas DONE!"