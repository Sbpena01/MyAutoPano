#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Example Job"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100
#SBATCH --error=SLURM_OUTPUT/slurm_train_%A.err
#SBATCH --output=SLURM_OUTPUT/slurm_train_%A.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

module load python/3.12.6/r3qjhak py-pip/24.0 

source ../panovenv/bin/activate

pip install -r requirements.txt

python -u Phase2/Code/Train.py --MiniBatchSize 64 --NumEpochs 50 --CheckPointPath Phase2/Checkpoints/
