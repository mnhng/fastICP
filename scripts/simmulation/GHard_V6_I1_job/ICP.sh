#!/usr/bin/bash
#SBATCH --array=0-99
source activate icp

mkdir -p cout/GHard_V6_I1
python launch_sim.py -i data/GHard_V6_I1/rep_$SLURM_ARRAY_TASK_ID \
    --method ICP \
    -o out/GHard_V6_I1/rep_$SLURM_ARRAY_TASK_ID
