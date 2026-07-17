#!/usr/bin/env bash
#SBATCH --job-name=llm_rsa_full_study
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --time=24:00:00
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

# Corrected GDV plus nested H1 and held-out H2 use the existing activation cache.
"$PYTHON" "$PROJECT_DIR/recompute_geometry.py"

# Forward-pass analyses, including the now-ready fixed-sentinel H5 design.
"$PYTHON" "$PROJECT_DIR/run_study.py" \
    --hypotheses H0 H3 H4 H5 \
    --words bank bark bat crane spring match pitch

"$PYTHON" "$PROJECT_DIR/plot_geometry_audit.py"
"$PYTHON" "$PROJECT_DIR/plot_semantic_layer_atlas.py"
"$PYTHON" "$PROJECT_DIR/plot_prior_matrix.py"
"$PYTHON" "$PROJECT_DIR/plot_h3_h4_audit.py"
"$PYTHON" "$PROJECT_DIR/plot_report_h1_h5.py"

"$PYTHON" "$PROJECT_DIR/validate_study_run.py" \
    --full --newer-than "$RUN_MARKER"

echo "Full corrected H0–H5 study run complete and validated"
