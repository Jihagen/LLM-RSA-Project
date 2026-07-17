#  A Homonym-Based Study of Semantic Disambiguation in Large Language Models

A reproducible study of whether transformer hidden states carry word-sense information in a way that can be read out across layers, token positions, and context conditions.

## What this repo does

- studies seven homonyms: bank, bark, bat, crane, spring, match, pitch
- excludes light because its two senses differ in part of speech and confound semantic vs. syntactic-category resolution
- evaluates hidden-state sense evidence with a signed centroid margin $M_l$ and its normalized form $M_l^{norm}$
- runs the H0–H5 analyses and regenerates the report figures and CSV outputs

## Key folders

- data/: stimuli and synthetic profiling data
- hypotheses/: one runner per hypothesis (H0–H5)
- experiments/: margin, adequacy, and GDV computation
- models/: model loading and activation extraction
- results/: generated outputs (gitignored)

## Quick start

```bash
cd llm_homonym_semantics
python run_study.py --hypotheses H1 H2
python recompute_geometry.py
python plot_geometry_audit.py
python plot_semantic_layer_atlas.py
python plot_prior_matrix.py
python plot_h3_h4_audit.py
python plot_report_h1_h5.py
```

For the full study on an HPC cluster, a Slurm launcher can be used locally, but it is intentionally not part of the public repository baseline.

## Reproduction checklist

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full step-by-step workflow.

1. Ensure the offline Hugging Face cache is present under the workspace hf_cache directory.
2. Use the project Python environment that has the required dependencies installed.
3. Confirm the stimulus data files exist in data/.
4. Run the study or the full Slurm job.
5. Regenerate the figures and validate the outputs with validate_study_run.py.
6. Inspect the CSVs and plots under results/study/ and results/study/figures/.

## Notes

- Results, figures, and run-specific artifacts are generated under results/ and are not tracked by git.
- The current public-facing analysis uses the seven-word study set and excludes light.
- The repository is intentionally focused on source code, stimuli, and reproducibility rather than HPC launchers or report-specific plots.
