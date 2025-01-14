#!/bin/bash
#SBATCH --job-name=gpt2_activations       # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=a100                  # Use A100 partition
#SBATCH --gres=gpu:a100:1                 # Request 1 A100 GPU
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Allocate 8 CPU cores

#SBATCH --time=06:00:00                   # 6 hours max runtime

# Load necessary modules
module load cuda/11.8

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_rsa_env

# Navigate to the project directory
cd $HOME/LLM-RSA-Project

# Run the Python script
python run_experiments.py