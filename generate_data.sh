source ./.venv/bin/activate

python Phase2/Code/DataGeneration.py --OutputPath Phase2/Data/Data_Generation/Train/ --ImagePath Phase2/Data/Train/ --NumImages 200 --PatchCount 50 
python Phase2/Code/DataGeneration.py --OutputPath Phase2/Data/Data_Generation/Val/ --ImagePath Phase2/Data/Val/ --NumImages 200 --PatchCount 50 
