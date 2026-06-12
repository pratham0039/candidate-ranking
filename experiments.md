# Experiments and Ablations

A log of what we tried, what we measured, and what we kept. All composites
are measured on our internal offline benchmark using the competition's
metric: `0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10`.

## Results timeline

| # | Configuration | Composite | Verdict |
|---|---------------|-----------|---------|
| 1 | Features only (no text similarity) | 0.700 | baseline |
| 2 | + TF-IDF similarity | 0.777 | kept |
| 3 | + MiniLM embedding similarity (default weights) | 0.767 | kept after tuning |
| 4 | + weight tuning, unconstrained | 0.986 | rejected (see exp A) |
| 5 | + weight tuning, bounded ranges | 0.982 | kept |
| 6 | + ownership-language factor, retuned | **0.988** | kept (final) |

Ranking runtime stayed 12-16 s on a laptop CPU throughout (limit: 5 min).

## Experiment A: unconstrained vs bounded weight search

Unconstrained random search reached 0.986 but drove the availability and
location exponents to ~0.05, effectively deleting both factors. Our offline
labels cannot fully observe how the hidden ground truth treats engagement
signals, and the job description explicitly instructs that dormant,
unresponsive candidates be down-weighted. Tuning a factor to zero because
the benchmark cannot see it is overfitting the benchmark, not improving the
ranker. We re-ran the search with bounded ranges (availability >= 0.3,
location >= 0.2, penalties >= 0.5) and accepted the 0.004 composite cost
for robustness. Kept: bounded.

## Experiment B: TF-IDF-dominant vs embedding-dominant blend

The tuned blend weighted TF-IDF 0.76 vs embeddings 0.25, which raised the
question whether the system was "mostly a keyword matcher". We flipped the
two weights and re-ranked:

| Blend | Composite | Top-100 composition |
|---|---|---|
| TF-IDF 0.76 / embed 0.25 (tuned) | 0.982 | strong profiles throughout |
| TF-IDF 0.25 / embed 0.76 (flipped) | 0.842 | 52 generic software engineers entered |

Embedding-dominant scoring let weaker generalist profiles climb: MiniLM
similarity separates "mentions ML" from "ships retrieval systems" less
sharply than rare-term TF-IDF does. Keyword-stuffing is not a counter-risk
because all text similarity runs over career-history descriptions, which
stuffed skill lists never reach, and stuffed profiles die at the title gate
regardless. Kept: tuned blend.

After adding the ownership factor (experiment D) and retuning, the blend
naturally rebalanced (TF-IDF 0.86 / embed 0.88) because ownership now does
the doer-vs-adjacent separation that rare terms were proxying for.

## Experiment C: title-gate false-negative audit

The title gate removes 68,821 of 100,000 candidates before scoring, so a
false negative there is unrecoverable. We audited every rejected profile
(not a sample) for strong retrieval/ML evidence in career descriptions
(embedding, retrieval, ranking system, FAISS, NDCG, fine-tuning, etc.):

- rejected profiles with strong ML evidence in career text: **0 / 68,821**

We also checked education-timeline anomalies as a possible additional
consistency signal and rejected the idea: 53% of the entire pool has an
education-to-career gap over 3 years and 11% has an implausibly short
bachelor's degree, uniformly across strong and weak profiles. A check that
fires on half the dataset has no discriminating power. Calibrating every
proposed check against the full pool before trusting it became a design
rule after this.

## Experiment D: ownership-language factor

Boundary analysis showed strong-skill candidates ranked below weaker but
highly-engaged ones near the bottom of the top-100. Reading those profiles:
the weaker ones described adjacency ("lighter weight than ranking systems
at FAANG", "production deployment was handled by the platform team") while
stronger ones described ownership ("owned the ranking layer", "designed the
relevance labeling pipeline"). We added a bounded ownership-vs-hedge ratio
computed over career descriptions and retuned:

- NDCG@50: 0.940 -> 0.967, MAP: 0.987, composite: 0.982 -> 0.988
- the top-100 now contains every candidate our benchmark rates highest

## Experiment E: manual boundary review

Before finalizing, we manually read every profile at ranks 80-120 (the
in/out boundary) as a skeptical second pass. This surfaced a profile whose
structured experience field said 2.7 years while its own summary text said
6.3 years. Prevalence analysis showed this contradiction pattern in 0.02%
of ordinary profiles but 7.3% of the strongest profile family, the same
rare-but-concentrated signature as the other planted inconsistencies. We
added it as consistency check #4; the affected profiles left the top-100
and the boundary refilled with internally consistent candidates.

## Experiment F: containerized reproduction

The full ranking step was reproduced from a fresh clone inside a
python:3.11-slim container with networking disabled and memory capped at
16 GB. Output matched the local submission byte-for-byte. Wall-clock was
49 s on container-local disk (the 275 s first attempt was Docker-for-Mac
bind-mount I/O overhead, worth knowing about when reproducing on a Mac).

## Experiment G: weight-perturbation sensitivity

The composite is measured against labels that share assumptions with the
system being tuned (the consistency gate and the benchmark's exclusions
overlap by construction), so the headline number alone could overstate
robustness. To quantify how much the ranking depends on the exact tuned
weights, we jittered every weight independently by +-20% across 200
samples and re-ranked:

- top-100 overlap with baseline: mean 97.8 / 100, minimum 91
- top-10 overlap with baseline: mean 9.7 / 10, minimum 8
- composite range under perturbation: 0.975 to 0.991

The shortlist is driven by the factor design, not by fragile weight
values. We also verified the gate/benchmark overlap directly: of the 482
profiles the consistency gate excludes, every exclusion from the two
strongest families (46 + 1) is independently flagged by the offline labels
as a planted inconsistency, and the remaining ~435 are weak-family
profiles whose exclusion costs nearly nothing on any plausible labeling.

## Experiment H: disqualifier prevalence audit

Following the design rule from experiment C, we measured prevalence before
implementing each remaining JD disqualifier across all 100,000 profiles:

- research-only careers (every role academic/research): 0
- profiles whose career text mentions LangChain at all: 0
- architect-titled current roles with no hands-on language: 0

All three checks would be dead code on this pool, so they are documented
here as calibrated decisions instead of implemented. The consulting-only
and vision/speech-only penalties did fire and are implemented.

## Experiment I: behavioral twins separate as designed

The dataset documentation names "behavioral twins" as a designed trap:
near-identical profiles whose platform signals differ. We located tier-4
family pairs sharing the same career fingerprint and yoe but divergent
engagement, and checked their fate in our ranking. Representative pairs:

| Active twin | Dormant twin | Outcome |
|---|---|---|
| resp 0.64, active May 10 | resp 0.41, active Apr 12 | rank 93 vs excluded from top-100 |
| resp 0.83, active Apr 21 | resp 0.41, active May 22 | rank 79 vs rank 100 |

The availability multiplier separates them in the expected direction while
keeping the dormant twin visible (heavily demoted, not erased), matching
the JD's instruction to down-weight unreachable candidates rather than
ignore skill.

## Rejected ideas

- **LLM re-ranking of the top ~300**: standard retrieve-cheap/re-rank-
  expensive pattern, but the ranking step must run with no network access,
  so any hosted-LLM call is out of scope by rule. LLM review was used only
  as an offline validation instrument during development.
- **Larger embedding model (BGE-large)**: experiment B showed the binding
  errors are not embedding-resolution errors; NDCG@10 was already saturated
  with MiniLM at 4x less precompute.
- **Education-timeline consistency checks**: see experiment C.
