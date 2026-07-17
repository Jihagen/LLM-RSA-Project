#!/usr/bin/env bash
#SBATCH --job-name=llm_rsa_h0_h3_h4_audit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/anvme/workspace/iwi5268h-llm_rsa/LLM-RSA-Project

set -euo pipefail
unset SLURM_EXPORT_ENV

PROJECT_DIR=/anvme/workspace/iwi5268h-llm_rsa/LLM-RSA-Project
PY_WORKSPACE=/anvme/workspace/iwi5268h-vision-transformers
CONDA_ENV_DIR="$PY_WORKSPACE/hpc_infrastructure/miniconda3/envs/vision-transformers"
PYTHON="$CONDA_ENV_DIR/bin/python"
BASE_DIR=/anvme/workspace/iwi5268h-llm_rsa

cd "$PROJECT_DIR"

export PATH="$CONDA_ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$BASE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export TORCH_HOME="$HF_HOME"

export OFFLOAD_DIR="$BASE_DIR/offload_dir_llm_rsa"
mkdir -p "$OFFLOAD_DIR" logs

export DEVROOT="$PY_WORKSPACE/dev"
export HF_DEPS_DIR="$DEVROOT/hf_deps"
export SCRATCH_TRANSFORMERS="$DEVROOT/transformers"
export PYTHONPATH="$HF_DEPS_DIR:$SCRATCH_TRANSFORMERS/src:${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DEVICE_MAP=auto

echo "Python: $PYTHON ($("$PYTHON" --version))"

"$PYTHON" "$PROJECT_DIR/validate_study_run.py" --preflight
RUN_MARKER="$PROJECT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.start"
touch "$RUN_MARKER"

"$PYTHON" "$PROJECT_DIR/run_study.py" \
    --hypotheses H0 H3 H4 \
    --words bank bark bat crane spring match pitch

# H5 is audited here but intentionally not run while directions and matched controls
# are incomplete. Human validation is not claimed or required; light is excluded.
"$PYTHON" -c \
    "from hypotheses.h5_garden_path import run_design_audit; print('H5 ready:', run_design_audit())"

"$PYTHON" "$PROJECT_DIR/plot_prior_matrix.py"
"$PYTHON" "$PROJECT_DIR/plot_h3_h4_audit.py"
"$PYTHON" "$PROJECT_DIR/validate_study_run.py" --newer-than "$RUN_MARKER"

echo "H0/H3/H4 rerun complete; H5 design audit written (no H5 model run)"
