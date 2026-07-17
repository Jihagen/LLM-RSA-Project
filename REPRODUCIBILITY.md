# Reproducibility guide

This document captures the minimal workflow needed to reproduce the study from a fresh checkout.

## 1. Environment

- Use the Python environment that has the required dependencies installed.
- Ensure the offline Hugging Face cache is available under the workspace-level hf_cache directory.
- Confirm that the project root is the llm_homonym_semantics directory.

## 2. Data dependencies

The following files are required for the study:

- data/paired_sentences.json
- data/garden_path_sentences.json
- data/synthetic_data_h2.pkl

## 3. Run the study

For the standard non-forward-pass workflow:

```bash
python run_study.py --hypotheses H1 H2
python recompute_geometry.py
```

For the full H0–H5 workflow on an HPC cluster, use a local Slurm launcher outside the repository baseline if desired.

## 4. Regenerate the figures

```bash
python plot_geometry_audit.py
python plot_semantic_layer_atlas.py
python plot_prior_matrix.py
python plot_h3_h4_audit.py
python plot_report_h1_h5.py
```

## 5. Validate outputs

```bash
python validate_study_run.py
```

## 6. Outputs

Generated outputs land under results/ and are expected to be local-only artifacts.
