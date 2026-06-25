# Redrob Candidate Ranker

Ranking system for the Intelligent Candidate Discovery & Ranking Challenge.
Reads a job description, scores a 100K candidate pool against it, and emits
a top-100 shortlist with per-candidate reasoning.

## Approach in one paragraph

The system ranks candidates the way a careful recruiter reads resumes.
A JD parser turns the job description into a structured spec: an experience
band, a location policy, must-have and nice-to-have competency groups, and
explicit rejection patterns. Candidates pass through two hard gates first: a
relevance gate (no software/data/ML work anywhere in the career history) and
a consistency gate (profiles whose dates and durations are internally
impossible are not trusted at any rank). Survivors are scored by a product
of interpretable factors: a skill-fit blend of competency-coverage evidence,
TF-IDF similarity and sentence-embedding similarity, computed over career
history text rather than self-reported skill lists, multiplied by an
experience-band trapezoid, a location factor, an engagement/availability
factor built from platform signals, and penalty factors for career patterns
the JD explicitly rejects. The reasoning column is generated from the same
factors that produced the score, so every claim in it traces to a profile
fact.

## Reproduce the submission

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. One-time precompute (downloads the MiniLM model, encodes the pool;
#    takes ~15 min on a laptop CPU, allowed to exceed the ranking budget)
python precompute.py --candidates path/to/candidates.jsonl

# 1b. SHORTCUT: skip the precompute by downloading the prebuilt artifacts
#     from the GitHub release instead:
#     https://github.com/pratham0039/candidate-ranking/releases/tag/artifacts-v1
#     Place embeddings.npz, tfidf.pkl, jd_embedding.npy into artifacts/

# 2. Ranking step (CPU only, no network, well under 5 minutes)
python rank.py --candidates path/to/candidates.jsonl \
               --jd data/job_description.md \
               --out submission.csv
```

`rank.py` needs only `artifacts/embeddings.npz`, `artifacts/tfidf.pkl`,
`artifacts/jd_embedding.npy` and `artifacts/weights.json`, all produced by
`precompute.py` and `tune_weights.py`. No network access is used during
ranking; the JD embedding falls back to the precomputed artifact when the
model cache is unavailable.

## Layout

```
src/jd_parser.py     job description -> structured spec
src/consistency.py   internal-consistency gate
src/features.py      interpretable scoring factors
precompute.py        one-time embeddings + TF-IDF artifacts
rank.py              the ranking step (single command, produces the CSV)
tune_weights.py      factor-weight search against our offline benchmark
sandbox/app.py       hosted demo: small sample in, ranked CSV out
```

## The JD's own 90-day plan, executed

The job description sketches the role's first 90 days. This repo is that
plan, run against the challenge pool:

- **Weeks 1-3, "audit what we currently have":** a full audit of the
  100K-candidate pool: family structure, planted inconsistencies, signal
  noise floors (experiments.md, experiments C and E).
- **Weeks 4-8, "ship a v2 ranking system":** the hybrid ranker in rank.py:
  evidence verification, TF-IDF + embedding relevance, availability
  multiplier, JD-derived penalties. Ranks 100K candidates in under a
  minute on CPU.
- **Weeks 9-12, "set up the evaluation infrastructure":** an offline
  benchmark, the official composite metric, a weight-tuning harness with
  bounded and minimax search, sensitivity analysis, and an automated
  reasoning audit (audit_reasoning.py).

## Reproduction inside the evaluation constraints

A Dockerfile is included that mirrors the evaluation environment. The full
ranking step was reproduced from a fresh clone inside it with
`--network none -m 16g`: identical output CSV, ~50 s wall-clock, <1 GB
peak memory. `make judge` runs ranking, format validation, and the
reasoning audit in one command.

## Design choices worth noting

- **Career text over skill lists.** Self-reported skills are cheap to
  inflate, so a skill only counts when the platform assessment or usage
  duration corroborates it, and primary evidence comes from what candidates
  actually did in their roles.
- **Consistency gate.** A profile claiming more experience than its career
  history can contain, or expert proficiency in skills never used, is
  excluded entirely, the same way a recruiter discards a resume that does
  not add up.
- **Soft experience band.** The JD says it seriously considers strong
  candidates outside the 5-9 band, so experience is a trapezoid factor, not
  a filter.
- **Engagement as a multiplier.** A perfect-on-paper candidate who never
  responds to recruiters is not hireable; availability multiplies the score
  rather than adding to it.
- **Weights are tuned, not guessed.** Factor exponents are optimized with
  random search against an offline benchmark we built during development,
  using the competition's own composite metric.
