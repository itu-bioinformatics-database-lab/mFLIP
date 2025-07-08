#!/bin/bash      
#SBATCH --account=bacan
#SBATCH --time=3-00:00:00
#SBATCH --job-name=describe_dataset
#SBATCH --partition=barbun
#SBATCH --ntasks=40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
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


python describe_dataset3.py
nvidia-smi
echo "describe_dataset DONE!"