#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Example Job"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --error=SLURM_OUTPUT/slurm_gen_data_%A_%a.err
#SBATCH --output=SLURM_OUTPUT/slurm_gen_data_%A_%a.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-2

IN_FOLDERS=("Phase2/Data/Data_Generation/Train/" "Phase2/Data/Data_Generation/Val/")
IM_PATHS=("Phase2/Data/Train/" "Phase2/Data/Val/")

mkdir -p ${IN_FOLDERS[$SLURM_ARRAY_TASK_ID-1]}"Patch_Stacks/" 
mkdir -p ${IN_FOLDERS[$SLURM_ARRAY_TASK_ID-1]}"Homographies/"

module load py-pip/24.0 

source panovenv2/bin/activate

pip install -r requirements.txt

python -u Phase2/Code/DataGeneration.py --OutputPath ${IN_FOLDERS[$SLURM_ARRAY_TASK_ID-1]} --ImagePath ${IM_PATHS[$SLURM_ARRAY_TASK_ID-1]} --PatchCount 64 --BatchSize 64
