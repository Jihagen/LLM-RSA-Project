#!/bin/bash
#SBATCH --job-name=test_cuda   # Job name
#SBATCH --output=logs/%x_%j.out                # Standard output log
#SBATCH --error=logs/%x_%j.err                 # Error log
#SBATCH --partition=a100                       # Use the a100 partition
#SBATCH --gres=gpu:a100:1                      # Request 1 GPU
#SBATCH --nodes=1                              # Use 1 node
#SBATCH --cpus-per-task=1                    # Use 16 CPU cores for faster data loading
#SBATCH --time=1-00:00:00                      # 1-day time limit
#SBATCH --export=NONE

# Load necessary modules
module load cuda/12.4                        # Load the CUDA module for 11.8
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_rsa_env 

# Navigate to the project directory
cd $HOME/LLM-RSA-Project

echo "TORCH_HOME: $TORCH_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Running nvidia-smi:"
nvidia-smi
echo "Checking Python CUDA availability:"

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('Current CUDA Device:', torch.cuda.current_device())
"


# Run the Python script
python test_cuda.py