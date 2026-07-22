#  Where Do Semantics Live? A Homonym-Based Study of Semantic Disambiguation in Large Language Models

A reproducible study of whether transformer hidden states carry word-sense information in a way that can be read out across layers, token positions, and context conditions.

## Project report

The full write-up is [Project_Report_LLM_homonym_disambiguation.pdf](Project_Report_LLM_homonym_disambiguation.pdf), covering methods, results, and discussion for all six analyses (H0–H5).

## Abstract

Transformer depth is sometimes interpreted as a processing hierarchy, raising the expectation that contextual meaning should become most clearly expressed within a reproducible semantic layer. We test this hypothesis using seven two-sense homonyms in four bidirectional encoders and four causal decoders, while separating three commonly conflated dimensions: layer depth, token position, and incrementally available context. Sense-conditioned representations are evaluated using held-out nearest-centroid margins and the Generalized Discrimination Value (GDV).

Contextual sense was widely decodable, but its strongest expression did not concentrate at one stable depth. Profiling-selected layers achieved 0.917 held-out adequacy, compared with 0.893 at the final layer, although the crossed model–word interval included zero and effects varied substantially across models and homonyms. Adequacy-based cross-word selection likewise achieved higher point adequacy than GDV-based selection (0.914 versus 0.892), while the two criteria frequently preferred different depths. Global class separation and reliable item-level nearest-centroid discrimination therefore captured different properties of the activation geometry.

Context-position analyses further showed that layer depth cannot be interpreted independently of causal context availability. In causal decoders, homonym-position adequacy was constrained to 0.500 when the resolving context followed the homonym. At the sentence-final period, after that context had become available, adequacy reached 0.761, and 67.9% of initially inadequate observations became locally decodable.

Incremental context-conflict experiments showed broad updating but incomplete geometric resolution. Resolving information moved a fixed sentinel toward the authored sense in 77.2% of observations, yet only 40.3% of initially primed states crossed into the resolved-sense region. Conflict endpoints also remained 0.140 lower than matched non-conflicting endpoints containing the same homonym and resolver, indicating a persistent cost of preceding conflicting context.

The results do not support a single, objective-independent semantic stage. Contextual sense is better characterised as a distributed, readout-relative, and path-dependent property of transformer representations. Behavioural similarity may therefore motivate comparisons with human language processing without implying a shared ordering of intermediate operations.

## Key results by hypothesis

- **H0 (baseline lean)** — Bare and neutrally-framed homonyms already lean toward one sense before any disambiguating context: mean absolute lean 0.339 across 56 model–word combinations (encoders 0.279, decoders 0.398), with some homonyms (e.g. *bank*, *bark*) showing a consistent direction across nearly all 8 models.
- **H1 (layer selection)** — Nested held-out layer selection reaches 0.917 adequacy vs. 0.893 at the final layer (2,240 outer folds), but the crossed model–word interval includes zero — the final layer is not uniformly worse, and the benefit is concentrated in a subset of decoders.
- **H2 (cross-word transfer)** — Adequacy-based cross-word layer selection (0.914) transfers better than GDV-based selection (0.892); the two criteria agree on the same layer in only 7/56 folds, showing that global cluster separation (GDV) and reliable item-level decoding (adequacy) capture different geometric properties.
- **H3 (causal availability)** — Moving the disambiguating clause from before to after the homonym barely changes encoder margins (0.0026) but collapses decoder margins almost to zero (effect 0.342), a decoder-vs-encoder interaction of 0.339 — direct evidence that causal masking constrains when sense information can appear.
- **H4 (sequence position)** — Decoder adequacy is fixed at 0.500 at the homonym (by construction) but recovers to 0.761 at the sentence-final period, with 67.9% of initially-inadequate cases becoming decodable — sense information can emerge later in the sequence even when unavailable at the ambiguous word itself.
- **H5 (incremental updating)** — A fixed sentinel readout moves toward the correct sense in 77.2% of observations after the resolver, but only 40.3% of initially-primed cases fully cross the decision boundary, and conflict endpoints remain 0.140 below matched non-conflicting controls — evidence of path-dependent, incomplete semantic revision.

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
