#!/bin/bash -l

#SBATCH --account=xx
#SBATCH --job-name="MITgcmInference"
#SBATCH --time=00:20:00
#SBATCH --ntasks=576
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mail-user=xx
#SBATCH --mail-type=end
# Script to run the simulation on the CSSC cluster

source activate swirl_toolbox

python main.py