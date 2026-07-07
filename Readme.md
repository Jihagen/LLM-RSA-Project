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
(`data/garden_path_sentences.json`, labelled with both a `primed_sense` and a
`correct_sense`), representations at the homonym position and the final position are
scored against both sense centroids. Expected signature: for encoders, a shift *across
layers* (early layers closer to the primed sense, late layers closer to the resolved
sense); for decoders, a shift *across token position* (homonym token stuck on the primed
sense, final token closer to resolved). This is explicitly exploratory — a directional
hint worth reporting, not a confirmatory claim of "backtracking."

---

## Repository layout

```
data/                    dataset construction & JSON stimuli
  paired_sentences.json        L/R context-position pairs (H3, H4)
  garden_path_sentences.json   primed/correct sense pairs (H5)
  synthetic_data_h1.pkl, ...   profiling sentences used to build sense centroids

models/                  model loading, encoder/decoder detection, activation extraction
experiments/
  adequacy.py                 M_l, centroid loading, per-layer adequacy profiles
  gdv_experiments.py           Generalized Discrimination Value (GDV) computation
  probing_experiments.py       linear probing classifiers over activations
hypothesis/
  h0_carrier_norming.py ... h5_garden_path.py     one runner per hypothesis
probing/                 standalone probing classifier utility
candidate_layers/        statistical layer-comparison notebooks/scripts (ANOVA, Tukey HSD)
utils/                   HPC runtime setup (offline HF cache, offload dir), file I/O
results/
  activations/{word}/{model}/       cached per-layer H5 activations + centroids
  {model}_gdv/                      GDV values and rank plots per model
  study/H{0..5}/{model}/            per-hypothesis CSV outputs + aggregate/summary CSVs

run_study.py             master CLI runner for H0–H5
run_h1.py, run_h2.py, run_experiments.py   legacy/standalone entry points
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
coverage; H0/H3/H4/H5 run on the four-model representative subset (DeBERTa-v3-large,
RoBERTa-large, Mistral-Nemo-Base-2407, Qwen2.5-3B) since they require fresh forward
passes over the paired/garden-path stimuli.

Outputs land under `results/study/H{n}/`, with one CSV per (model, word) and an
aggregate/summary CSV per hypothesis for cross-model comparison.
