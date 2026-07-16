# LLM–EEG Master Project — Neuroscience Lab, FAU

## Neural and Artificial Correlates in Language Processing: A Comparative Study of LLMs and EEG data

**Research question:** Do large language models (LLMs) display activation patterns that
reflect the regional specialisation observed in the brain's language processing areas?

This repository holds the LLM side of that comparison: a word-sense-disambiguation (WSD)
study, using homonyms (*bank, bark, bat, crane, spring, match, light, pitch*) as the test
case for asking a more basic question first — **when can a hidden state be trusted as a
word-in-context representation at all?** The homonym/context manipulation is the tool;
the target is the representation itself: which layer, which token position, and which
architecture (encoder vs. decoder) actually carry the resolved sense.

Eight models are covered, spanning encoder and decoder families and two size tiers:

| Type | Models |
|---|---|
| Encoder | `ModernBERT-large`, `DeBERTa-v3-large`, `RoBERTa-large`, `XLM-RoBERTa-large` |
| Decoder | `Qwen2.5-3B`, `Qwen2.5-7B`, `Mistral-Nemo-Base-2407`, `OLMo-2-1124-7B` |

---

## The core measure: sense margin M_l

Every hypothesis (H1–H5) reduces to the same quantity, computed per layer *l* from a
hidden state *h_l* and two sense centroids *c_correct*, *c_wrong* (mean activations of
sentences using the intended vs. the other sense of the homonym):

```
M_l = ||h_l - c_wrong|| - ||h_l - c_correct||
```

- `M_l > 0` → representation sits closer to the correct sense (adequate)
- `M_l ≈ 0` → ambiguous, on the decision boundary
- `M_l < 0` → representation sits closer to the wrong sense (inadequate)

A representation is called **adequate** when `M_l > epsilon` (default `epsilon = 0`).
Everything below is a different way of slicing where and when `M_l` is (and isn't)
positive.

Raw margin magnitude is not comparable across layers or architectures. Reported
cross-layer magnitudes therefore use the inter-centroid-normalized value:

```
M_l_norm = M_l / ||c_correct - c_wrong||
```

For Euclidean distances, the reverse triangle inequality guarantees
`-1 <= M_l_norm <= 1`; the implementation checks this invariant at runtime. Output
columns explicitly distinguish `*_raw` from `*_norm`.

---

## Hypotheses

| # | Question | Status |
|---|---|---|
| H0 | Is the bare word / carrier sentence already sense-biased before context is added? | Prerequisite control |
| H1 | Is the last layer always the best layer for sense adequacy? | Core |
| H2 | Does GDV-selected layer generalise to held-out words? | Core |
| H3 | Do decoders lose adequacy when disambiguating context comes *after* the homonym? | Core |
| H4 | Is sentence sense decodable at the homonym and final token positions? | Descriptive |
| H5 | Does a fixed readout change as progressively longer context-conflict prefixes arrive? | Exploratory; balanced 98-item rerun complete |

### H0 — Carrier Norming
*Not a theory test — a control.* Before attributing any effect in H3 to "context," we
need to know the word and its neutral carrier sentence ("The bank was unstable.") aren't
already biased toward one sense. H0 computes `M_l` for the word alone and for the carrier
alone; both should be near zero. Anything else means that word/carrier pair needs to be
flagged or excluded, since later "context effects" would really be pre-existing bias.

### H1 — Layer Adequacy
Many analyses default to the final hidden layer. H1 profiles each layer using
leave-one-sentence-out centroids. Its best-versus-last comparison is nested: for every
outer held-out sentence, the best layer is selected using only the remaining sentences,
then both that layer and the last layer are evaluated on the untouched sentence. This
removes centroid self-inclusion and the optimism from selecting and evaluating a maximum
on the same observations.

### H2 — GDV Generalisation
H1 shows layers differ; H2 asks whether the label-dependent, probe-free GDV geometry
score can select a useful layer for a held-out word. In each leave-one-word-out fold,
three fair strategies are compared: GDV selected on the other seven words, supervised
adequacy selected on the other seven words, and the final layer. The held-out word is
always scored with leave-one-sentence-out centroids. Its own best layer is retained only
as an explicitly optimistic oracle ceiling.

### H3 — Right-Context Vulnerability
This is where architecture should matter. A decoder's hidden state at the homonym token
is causally blind to anything after it — so if the disambiguating clause comes *after*
the homonym (R condition), the decoder's homonym-token representation should still be
adequate for the primed sense, not the true one. An encoder's homonym token has
bidirectional access, so it shouldn't care whether the clue is before (L) or after (R).
Using paired L/R sentences (`data/paired_sentences.json`), `M_l` is computed at the
homonym token for both conditions. Decoder R=50% is a deterministic manipulation check:
mirrored sense labels score the same causal-prefix state with opposite signs. Inference
therefore targets the paired L-minus-R change and the decoder-minus-encoder interaction,
with repeated sentences first aggregated within word.

### Keep three axes separate

- **Layer depth** is a succession of transformations inside one forward pass. For an
  encoder, the homonym can attend to the full supplied input from its first attention
  layer; for a causal decoder, deeper layers never reveal tokens to the homonym that
  occur to its right.
- **Sequence position** compares different token states within one supplied sequence.
  Later positions have different causal context in a decoder, but also different token
  identities and positional distributions.
- **Incremental reading time** reruns a model on progressively longer prefixes. Claims
  about commitment, updating, or garden-path resolution require this third design.

### H4 — Token-position sense decodability
H4 is a sequence-position analysis from one completed forward pass, not a time course.
The homonym state is scored against homonym-position profiling centroids, while the
final state is scored primarily against final-position profiling centroids. This asks
the weaker, defensible question: is sentence sense decodable at each position in its
own local geometry? Scoring the final state against homonym centroids is retained only
as a cross-position transfer diagnostic; failure can reflect token/position
distribution shift and does not identify a new representational basis.

The full 2x2 table is saved. Headline summaries condition on the starting state:
`P(final adequate | target inadequate)` and the reverse transition
`P(final inadequate | target adequate)`, with denominators and Wilson intervals. The
raw proportion in one transition cell is not called a recovery rate.

### H5 — Incremental fixed-sentinel updating (exploratory)
The original homonym→resolution-word line changed token identity and position; it could
therefore reflect the lexical semantics of the resolver itself (for example, *wings*)
rather than revision. That output is obsolete.

The corrected H5 reruns three progressively longer inputs: context before the homonym,
context through the homonym, and context through the annotated resolver. Every input
ends with the same separate-paragraph readout token (`probe`). Its state is scored
against probe-position centroids learned from independent disambiguated profiling
sentences. The pre-homonym stage is a context-bias baseline; initial commitment is
assessed after the homonym. The primary estimand is the paired continuous
resolution-minus-homonym change in normalized correct-sense margin. A sign transition
is reported only conditional on a primed homonym-stage state. Resolver-alone and
matched non-garden control scores separate contextual updating from lexical resolver
effects.

The current stimulus set is balanced in both primed→correct directions for every
eligible homonym, all 98 items have matched controls, and the structural audit passes.
External human norms will not be collected, so these are described as authored
context-conflict stimuli rather than human-validated garden paths. `light` is
intentionally excluded because its two
study senses differ in part of speech, confounding syntactic-category and semantic
resolution. `run_h5` writes a design audit and stops before
loading a model unless `--allow-incomplete-h5` is explicitly supplied. Such override
outputs are labelled `exploratory_incomplete_design`.

---

## Repository layout

```
data/
  paired_sentences.json        L/R context-position pairs (H0, H3, H4) — tracked in git
  garden_path_sentences.json   primed/correct sense pairs + resolution_word (H5) — tracked in git
  synthetic_data_h2.pkl        profiling sentences used to build sense centroids (H1, H2) — tracked in git
  synthetic_data_preparation.py    flatten_dataframe() — the only actively-used helper here
  synthetic/synthetic_datageneration.py   one-off LLM-based sentence-generation script (needs OPENAI_API_KEY)
  inspect_paired_sentences.ipynb, inspect_and_build_dataset.ipynb   stimulus-curation notebooks (provenance, not run by the pipeline)

models/                  model loading, encoder/decoder detection, activation extraction
experiments/
  adequacy.py                  M_l / symmetric per-sentence margin, centroid loading, per-layer adequacy profiles
  gdv_experiments.py            Generalized Discrimination Value (GDV) computation
hypotheses/
  h0_carrier_norming.py ... h5_garden_path.py     one runner per hypothesis
utils/                   HPC runtime setup (offline HF cache, offload dir), file I/O
results/                 gitignored — regenerated by the commands below
  activations/{word}/{model}/       cached per-layer H5 activations + centroids
  {model}_gdv/                      GDV values and rank plots per model
  study/H{0..5}/{model}/            per-hypothesis CSV outputs + aggregate/summary CSVs
  study/study_results.ipynb         executed analysis notebook (the write-up)

run_study.py             master CLI runner for H0–H5 (reruns scoring against cached activations)
run_h2.py                (re)generates the activation cache + GDV values from data/synthetic_data_h2.pkl
```

## Running the study

```bash
# Default: H1 + H2 only (no forward passes — reuses cached activations)
python run_study.py

# Specific hypotheses
python run_study.py --hypotheses H3 H4 H5

# Restrict to a model subset (aliases: deberta, roberta, xlm, modernbert,
# qwen3b, qwen7b, mistral, olmo)
python run_study.py --hypotheses H1 --models deberta roberta

# Restrict to specific homonyms
python run_study.py --hypotheses H1 H2 --words bank bark

# Recompute corrected GDV, nested H1, and held-out H2 from cached H5 files only
python recompute_geometry.py

# Audit H5 stimuli without loading a model or using a GPU
python -c "from hypotheses.h5_garden_path import run_design_audit; print(run_design_audit())"

# Explicitly run the incomplete H5 stimuli for development only
python run_study.py --hypotheses H5 --allow-incomplete-h5

# Regenerate the audited Figure 4 replacement
python plot_geometry_audit.py

# Regenerate the agreement-first model x homonym layer map
python plot_semantic_layer_atlas.py

# Regenerate signed bare-word and neutral-carrier priors
python plot_prior_matrix.py

# Inspect the context stages without loading a model
python context_revelation_trajectory.py --dry-run --words bank

# Fresh H0/H3/H4 inference at the corrected H1-selected layers; H5 is audited
# but intentionally not run (submit explicitly)
sbatch run_h0_h3_h4_audit.slurm
```

Data prerequisites per hypothesis:
- **H0, H3, H4:** `data/paired_sentences.json`
- **H1, H2:** cached activations in `results/activations/{word}/{model}/`
- **H5:** `data/garden_path_sentences.json` plus `data/synthetic_data_h2.pkl`;
  the model-internal analysis additionally requires balanced directions and a matched
  control for every item
- **H2:** additionally needs `results/{model}_gdv/gdv_values_{word}.csv` for every word

All eight profiling words (bank, bark, bat, crane, spring, match, light, pitch) and all
eight models currently have cached activations and GDV values, so H1/H2 run at full
coverage; H0/H3/H4/H5 also default to the full eight-model set (`H3_MODELS` in
`hypotheses/h3_context_position.py`), requiring a fresh forward pass per model over the
paired/garden-path stimuli rather than reusing cached activations.

Outputs land under `results/study/H{n}/`, with one CSV per (model, word) and an
aggregate/summary CSV per hypothesis for cross-model comparison.

---

## Current geometry results (2026-07-14, full 8-model x 8-word run)

H1/H2 were regenerated entirely from cached activations after correcting GDV and adding
held-out evaluation. H3's existing sentence-level outputs were reanalysed with the valid
paired estimand. H4 and H5 have now been redefined, but neither has a new forward-pass
result yet. Existing H4 recovery labels and H5 cross-token trajectories are obsolete.

| # | Verdict |
|---|---|
| H0 | **Design corrected; rerun pending.** Mirrored `M`/`-M` label rows are collapsed to five independent carriers per word. Signed direction, normalized strength, bare-word shift, and cross-carrier consistency replace the old doubled count and confirmatory `|M| > 0.3` label. |
| H1 | **Suggestive, heterogeneous best-layer benefit.** Nested selection reaches 92.32% versus 89.65% at the last layer (+2.66 points; reproducible crossed-bootstrap interval -0.34 to +6.41). The gain is concentrated in decoders (+4.78 points); several individual models show no benefit or a small reversal. |
| H2 | **GDV does not improve on the final layer overall.** GDV reaches 89.43%, the last layer 89.65% (-0.22 points; reproducible crossed-bootstrap interval -5.43 to +4.25), supervised cross-word selection 92.41%, and the held-out oracle ceiling 93.70%. The supervised advantage is clearest for decoders. This supports objective-dependent layer quality, not a claim that GDV is generally useless. |
| H3 | **The paired architecture effect is the result; decoder R=50% is a sanity check.** Existing outputs give mean normalized L-R effects of 0.3053 for decoders and 0.0035 for encoders; interaction 0.3018, crossed model-by-word interval 0.2015 to 0.4057. Rerun pending at corrected H1 layers. |
| H4 | **Design corrected; rerun pending.** It now estimates within-position decodability and conditional transitions. Cross-position centroid transfer is diagnostic only, not recovery or basis change. |
| H5 | **Balanced rerun complete.** The fixed-sentinel analysis contains all 784 model--item observations from 98 balanced context-conflict items, with matched controls and isolated-resolver baselines. The primary continuous update is positive, while conditional resolution remains incomplete. `light` is a documented cross-POS exclusion. No claim of human-validated garden-path processing is made. |

### Data-quality/correctness fixes behind this run
1. **Activation cache:** the profiling-sentence target-word finder matched substrings
   without word boundaries (e.g. "bat" inside "batsman"/"Bats", "light" inside
   "Sunlight"), corrupting sense centroids for `bat`, `crane`, `match`, `light`. Fixed in
   `models.find_target_span`; the whole activation cache was regenerated.
2. **Scoring (the most consequential fix):** H1/H2/H3/H4 scored every sentence in a
   word's profiling/paired/R-condition batch — balanced 50/50 across both senses —
   against a single **fixed** sense-0-is-correct centroid pair. For the true sense-1 half
   of any batch this silently used the wrong centroid as "correct," and because the two
   halves cancel, `frac_adequate` converges toward ~50% *by construction*, independent of
   true separability. Verified directly: DeBERTa on `bank` showed 100% true classification
   accuracy while the old code reported exactly 0.5. Fixed with
   `experiments.adequacy.symmetric_adequacy_margins`, which scores each sentence against
   its own true sense.
3. **Final-token position (legacy H4/H5):** `get_dual_position_activations` took the last
   non-padding token as the "final" position, which for BERT-family encoders is `[SEP]` —
   a structural marker, not real content. Fixed to skip trailing special tokens. This did
   *not* rescue a hidden encoder effect; the null result got more uniform, not less (see
   the H4 verdict above) — informative in its own right.
4. **GDV equation:** class and class-pair means were divided by their class-count factors
   a second time, and the published `1/2` feature scaling was omitted. Corrected and
   covered by equation and invariance tests.
5. **Centroid leakage and layer search:** H1/H2 now use leave-one-sentence-out centroids;
   H1's best-versus-last estimate nests layer selection inside each outer sentence fold.
6. **H2 baselines:** added supervised cross-word selection and relabelled the held-out
   best layer as an oracle rather than a fair competitor.
7. **H3 inference:** removed the binomial tests against 0.5 and replaced them with paired
   L-R effects and a crossed model-by-word architecture interaction.
8. **Figure 4:** replaced the arbitrary-axis projection with genuine PCA panels,
   displayed-point centroids and boundary. Sentence decision scores and full layer
   curves now have separate, single-purpose figures; the curves use separate axes
   rather than a dual-axis plot.
9. **Word-level heterogeneity:** replaced aggregate-first layer summaries with an
   agreement-first model x homonym map. Dark cells mean GDV and `M_l` select the same
   layer; light cells print the `M_l` to GDV layer change directly.
10. **Signed priors:** added separate bare-word and neutral-carrier matrices. Hue is
    prior direction, saturation is normalized strength, and marker fill shows
    cross-carrier directional consistency.

If you see numbers that do not match the draft PDF, use the CSVs generated by
`recompute_geometry.py`. The draft's H1/H2 tables and H3 chance tests are obsolete.

### Known open items
- Run corrected H4 after GPU work is authorized; interpret only local decodability and
  the two conditional transition probabilities.
- The balanced fixed-sentinel H5 design has been run successfully. Directions and matched controls are
  complete; retain `light` as a documented cross-POS exclusion. Optional human-rating
  fields remain supported but are not required and will not be collected.
- The earlier context-revelation runner remains a visualization prototype for paired
  ordinary sentences; it is not evidence of garden-path revision and must not be used
  as the H5 analysis.
- Raw `M_l` magnitude is not comparable across architectures or even across layers within
  a model: decoder-only LLMs develop a handful of very-high-magnitude ("massive
  activation") hidden dimensions concentrated in later layers, and decoders' best layers
  are usually late — so raw M_l can be 1–2 orders of magnitude larger for decoders on some
  words (e.g. Qwen2.5-3B `spring`: 871, `bat`: 545) for reasons unrelated to sense
  adequacy. Sign and `frac_adequate` are robust to this; raw means are not. Normalized
margins are now emitted explicitly and checked against their mathematical bounds.
Crossed-bootstrap intervals for the H1 and H2 comparisons are reproducibly regenerated
as `results/study/geometry_inference.csv` by `recompute_geometry.py`.
- Encoder/decoder averages are descriptive patterns in these eight selected models, not
  causal estimates of the attention mask: the groups also differ in size, objectives,
  data, tokenisers, depth, positional encoding, and training recipe. H3's paired target-
  token manipulation is the one analysis that directly identifies causal availability.
- Profiling-sentence cue position has not yet been annotated or balanced. Decoder H1
  performance can therefore reflect how often the decisive cue falls to the right of
  the homonym, not merely where sense-conditioned separation peaks in depth.

## Visual encoding system

Colors have fixed roles and are derived from the original report palette rather than
Matplotlib defaults. Sense 0 is violet and sense 1 amber. Encoder/decoder comparisons
use dark teal and dusty mauve. `M_l` is charcoal and GDV plum. Agreement is dark sage;
disagreement is warm stone. Prior matrices use amber -> neutral -> violet so hue means
direction and saturation means strength. Context position uses line/marker fill rather
than introducing another pair of colors. Outcome states retain separate green/grey/
muted-brick roles. All definitions live in `visual_style.py`.
