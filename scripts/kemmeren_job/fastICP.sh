#!/usr/bin/bash
#SBATCH --array=0-999
#SBATCH --mem=8G
source activate icp

./run_kemmeren.py --method fastICP --out kemmeren/fastICP/ --target $SLURM_ARRAY_TASK_ID
./run_kemmeren.py --method fastICP --out kemmeren/fastICP/ --target $((SLURM_ARRAY_TASK_ID + 1000))
./run_kemmeren.py --method fastICP --out kemmeren/fastICP/ --target $((SLURM_ARRAY_TASK_ID + 2000))
./run_kemmeren.py --method fastICP --out kemmeren/fastICP/ --target $((SLURM_ARRAY_TASK_ID + 3000))
./run_kemmeren.py --method fastICP --out kemmeren/fastICP/ --target $((SLURM_ARRAY_TASK_ID + 4000))
./run_kemmeren.py --method fastICP --out kemmeren/fastICP/ --target $((SLURM_ARRAY_TASK_ID + 5000))
./run_kemmeren.py --method fastICP --out kemmeren/fastICP/ --target $((SLURM_ARRAY_TASK_ID + 6000))
