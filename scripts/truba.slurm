#!/bin/bash

#SBATCH --account=bacan
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=00:15:00
#SBATCH --time=3-00:00:00
#SBATCH --job-name=test

#SBATCH --partition=palamut-cuda
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

###SBATCH --mal-user= canb22@itu.edu.tr
###SBATCH --mail-type=BEGIN,END,FAIL
###SBATCH --mail-type=ALL

### Load modules

module purge
module load apps/truba-ai/gpu-2024.0
module load lib/cuda/12.4
# module load comp/python/ai-tools-kolyoz-1.0

source ../venv/bin/activate

echo "We have the modules: $(module list 2>&1)" > ${SLURM_JOB_ID}.info

### jobs
nvidia-smi

exit