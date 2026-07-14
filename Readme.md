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

---

## Hypotheses

| # | Question | Status |
|---|---|---|
| H0 | Is the bare word / carrier sentence already sense-biased before context is added? | Prerequisite control |
| H1 | Is the last layer always the best layer for sense adequacy? | Core |
| H2 | Does GDV-selected layer generalise to held-out words? | Core |
| H3 | Do decoders lose adequacy when disambiguating context comes *after* the homonym? | Core |
| H4 | If so, does a later token position recover it? | Core |
| H5 | Do garden-path sentences leave a geometric trace of initial (wrong) commitment and later revision? | Exploratory |

### H0 — Carrier Norming
*Not a theory test — a control.* Before attributing any effect in H3 to "context," we
need to know the word and its neutral carrier sentence ("The bank was unstable.") aren't
already biased toward one sense. H0 computes `M_l` for the word alone and for the carrier
alone; both should be near zero. Anything else means that word/carrier pair needs to be
flagged or excluded, since later "context effects" would really be pre-existing bias.

### H1 — Layer Adequacy
Many analyses default to the final hidden layer. H1 tests whether that default is
justified by computing `M_l` and fraction-adequate per layer, per (model, word), from
cached activations. Expectation: adequacy peaks at a model-specific layer, not
necessarily the last one — motivating explicit layer selection rather than "just take
the last layer."

### H2 — GDV Generalisation
H1 shows layers differ; H2 asks whether a cheap, unsupervised geometry score (GDV) can
*predict* the good layer without touching the test word. Using leave-one-word-out
evaluation, the GDV-best layer (chosen from the other words) is compared against the last
layer and the empirically-best adequacy layer on the held-out word. If GDV-best tracks
adequacy-best, GDV becomes a practical layer-selection method rather than just a
descriptive statistic.

### H3 — Right-Context Vulnerability
This is where architecture should matter. A decoder's hidden state at the homonym token
is causally blind to anything after it — so if the disambiguating clause comes *after*
the homonym (R condition), the decoder's homonym-token representation should still be
adequate for the primed sense, not the true one. An encoder's homonym token has
bidirectional access, so it shouldn't care whether the clue is before (L) or after (R).
Using paired L/R sentences (`data/paired_sentences.json`), `M_l` is computed at the
homonym token for both conditions. Expectation: encoders stay adequate in L and R;
decoders drop in R.

### H4 — Decoder Recovery / Token-Position Dissociation
H3 asks about the homonym token only. H4 asks the follow-up: even if the decoder's
homonym-token state is stuck on the primed sense, does the model recover by the *final*
token, once it has actually seen the disambiguating clause? Using one forward pass per
R-condition sentence, both the homonym-token and final-token representations are scored
against the same centroids. **Dissociation** = homonym token inadequate, final token
adequate. This separates "the model produces the right answer downstream" from "the
hidden state at the ambiguous word encodes the right sense" — two claims that are often
conflated. Encoders are expected to show little dissociation (their homonym token is
already full-context).

### H5 — Garden-Path / Representational Revision (exploratory)
A bridge to the psycholinguistic notion of commitment-and-revision: humans often commit
to an initial reading and then revise it when later context contradicts it. H5 asks
whether models show a geometric analogue. Using curated garden-path sentences
(`data/garden_path_sentences.json`, labelled with a `primed_sense`, a `correct_sense`,
and a `resolution_word`), representations at the homonym position and at the annotated
resolution word are scored against the homonym-position sense centroids. Expected
signature: for encoders, a shift *across layers* (early layers closer to the primed
sense, late layers closer to the resolved sense); for decoders, a shift *across token
position* (homonym token stuck on the primed sense, resolution-word token closer to
resolved). This is explicitly exploratory — a directional hint worth reporting, not a
confirmatory claim of "backtracking."

Scoring is done at the specific word that carries the disambiguation
(`resolution_word`), not the sentence's last token. An earlier version scored the last
non-special token as a proxy for "has seen the disambiguating clause," which silently
breaks whenever a sentence happens to end on a neutral filler or, worse, a word
associated with the *primed* sense (e.g. a bird-revealing crane sentence ending on
"...flew over the scaffolding" — a construction noun). Annotating the actual resolving
word removes that dependency on how each stimulus happens to end, and removes the need
for a separate final-position centroid baseline the earlier design used to guard
against exactly that failure mode.

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
```

Data prerequisites per hypothesis:
- **H0, H3, H4:** `data/paired_sentences.json`
- **H1, H2:** cached activations in `results/activations/{word}/{model}/`
- **H5:** `data/garden_path_sentences.json`
- **H2:** additionally needs `results/{model}_gdv/gdv_values_{word}.csv` for every word

All eight profiling words (bank, bark, bat, crane, spring, match, light, pitch) and all
eight models currently have cached activations and GDV values, so H1/H2 run at full
coverage; H0/H3/H4/H5 also default to the full eight-model set (`H3_MODELS` in
`hypotheses/h3_context_position.py`), requiring a fresh forward pass per model over the
paired/garden-path stimuli rather than reusing cached activations.

Outputs land under `results/study/H{n}/`, with one CSV per (model, word) and an
aggregate/summary CSV per hypothesis for cross-model comparison.

---

## Current results (2026-07-08, fully corrected pipeline, full 8-word run)

All six hypotheses have been run to completion on the full 8-homonym set after three
rounds of data-quality/correctness fixes (below), and the analysis is written up in
[`results/study/study_results.ipynb`](results/study/study_results.ipynb). Headline findings:

| # | Verdict |
|---|---|
| H0 | **Holds.** No carrier sentence is fully neutral. Raw bias magnitude isn't comparable across architectures — see the rogue-dimension note below. |
| H1 | **Holds, with a real architectural difference.** Encoders separate senses almost perfectly (97.5% mean adequacy) in upper-middle layers; decoders do well but more variably (76.9% mean, from 59.7% for Qwen2.5-7B to 87.6% for OLMo) and lean much more on their final layers (65.6% of folds vs. 31.3% for encoders). |
| H2 | **Falsified.** GDV predicts the adequacy-best layer in only ~12.5% of leave-one-word-out folds — no model is a genuine exception. |
| H3 | **Confirmed, cleanly, and statistically significant.** Decoders drop from 76.3% to exactly 50.0% (chance) adequacy when the disambiguating clause moves after the homonym, in 8/8 words for *both* decoder models (sign-test p = 0.008 each). Encoders are unaffected (4/8 words — chance). The cleanest, most decisive result in the study. |
| H4 | **Real but modest for decoders; the encoder half is a genuine negative result, not a bug.** Decoder target-token adequacy collapses to exactly chance (0.500) and only partially recovers at the final token (0.550, dissociation rate 28.1%). For encoders, `frac_adeq_final` is **exactly 0.500 in 16 of 16 (model, word) cells** — too precise to be noise. It means an encoder's true final content token sits almost exactly on the sense boundary regardless of word/model: a structural mismatch between what that token represents and what homonym-position centroids detect, not something a token-position fix can resolve. |
| H5 | **Directionally consistent across architectures, still exploratory.** Both encoders and decoders show target→final improvement in 68.75% of (model, word) cells, unchanged by the round-3 fix below — a genuine cross-architecture signal — but only 2–3 garden-path sentences per cell, so treat as suggestive, not resolved. |

### Data-quality/correctness fixes behind this run (three rounds, all required)
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
3. **Final-token position (H4/H5 only):** `get_dual_position_activations` took the last
   non-padding token as the "final" position, which for BERT-family encoders is `[SEP]` —
   a structural marker, not real content. Fixed to skip trailing special tokens. This did
   *not* rescue a hidden encoder effect; the null result got more uniform, not less (see
   the H4 verdict above) — informative in its own right.

If you see numbers that don't match an older export of this repo, trust the current run —
the table above and the notebook reflect all three fixes.

### Known open items
- **Redesign H4's encoder half** around final-position-specific centroids (or drop it) —
  this is now a design question, not a bug to patch.
- H5 now runs 6–7 garden-path sentences per word (up from 2–3) across the full 8-model
  set, and scores the annotated `resolution_word` position instead of the sentence-final
  token (see H5 section above) — the aggregate direction should be re-checked against
  this rerun before being treated as confirmed.
- Raw `M_l` magnitude is not comparable across architectures or even across layers within
  a model: decoder-only LLMs develop a handful of very-high-magnitude ("massive
  activation") hidden dimensions concentrated in later layers, and decoders' best layers
  are usually late — so raw M_l can be 1–2 orders of magnitude larger for decoders on some
  words (e.g. Qwen2.5-3B `spring`: 871, `bat`: 545) for reasons unrelated to sense
  adequacy. Sign and `frac_adequate` (or medians, where used above) are robust to this;
  raw means are not — a normalized/robust distance metric would be a good next step.
