#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Example Job"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --error=slurm_gen_data_%A.err
#SBATCH --output=slurm_gen_data_%A.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

module load python/3.12.6/r3qjhak py-pip/24.0 

source ../panovenv/bin/activate

pip install -r requirements.txt

python -u Phase2/Code/DataGeneration.py --OutputPath Phase2/Data/Data_Generation/Train/ --ImagePath Phase2/Data/Train/ --NumImages 200 --PatchCount 50 
python -u Phase2/Code/DataGeneration.py --OutputPath Phase2/Data/Data_Generation/Val/ --ImagePath Phase2/Data/Val/ --NumImages 200 --PatchCount 50 
