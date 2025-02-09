#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Training Job"
#SBATCH -A rbe549
#SBATCH -p academic
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=SLURM_OUTPUT/slurm_train_%A.err
#SBATCH --output=SLURM_OUTPUT/slurm_train_%A.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

module load py-pip/24.0 

source panovenv2/bin/activate

pip install -r requirements.txt

python -u Phase2/Code/Train.py --ModelType Unsup --MiniBatchSize 64 --NumEpochs 50 --CheckPointPath Phase2/Checkpoints_Unsup/
