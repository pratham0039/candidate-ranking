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

## Rejected ideas

- **LLM re-ranking of the top ~300**: standard retrieve-cheap/re-rank-
  expensive pattern, but the ranking step must run with no network access,
  so any hosted-LLM call is out of scope by rule. LLM review was used only
  as an offline validation instrument during development.
- **Larger embedding model (BGE-large)**: experiment B showed the binding
  errors are not embedding-resolution errors; NDCG@10 was already saturated
  with MiniLM at 4x less precompute.
- **Education-timeline consistency checks**: see experiment C.
