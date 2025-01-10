#!/bin/bash

#SBATCH --job-name=train_stage1    # Name of the job
#SBATCH --cpus-per-task=6          # Number of CPU cores
#SBATCH --mem=100G                 # Memory allocation
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --nodelist=ada06           # Specific node
#SBATCH --time=4-00:00:00          # Maximum runtime (5 days)
#SBATCH --output=train_stage1_output.log # Output file for the job
#SBATCH --error=train_stage1_error.log   # Error file for the job

# Load necessary modules or activate environment
# Example: module load python/3.8
# source activate myenv

# Run the Python script
conda activate hyper
python main_1stage.py --config-name=train_1stage_parallel_big_3_nn