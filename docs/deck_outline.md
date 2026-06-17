# Submission Deck — Outline & Speaker Context

A 12-slide deck (export to PDF) for the Redrob Intelligent Candidate
Discovery & Ranking Challenge. This file is the source of truth for the
deck's content and the talking points behind each slide.

**Audience:** Redrob engineers — the people who wrote the JD and will run
Stages 3–5. Tone: confident, specific, honest about limitations, no fluff.

**Every number below is live as of 2026-06-17**, pulled from `submission.csv`,
`artifacts/weights.json`, and `eval_harness.py`. If the pipeline is retuned,
re-verify Slide 6 (worked example) and Slides 8–9 (metrics) before exporting.

---

## Background the deck rests on (context for the presenter)

**What the challenge is.** Rank 100,000 synthetic candidate profiles
against one job description ("Senior AI Engineer, Founding Team" at Redrob,
a Series-A HR-tech company). Submit a top-100 CSV with a 1–2 sentence
reasoning per candidate. Judged by a hidden composite metric, then code
reproduction, then manual review, then a defend-your-work interview.

**Why it's hard.** The dataset is adversarial by design. The organizers
planted keyword stuffers, ~80 internally-impossible "honeypots" (>10% in
your top-100 disqualifies you), strong candidates who use zero buzzwords,
and "behavioral twins" (identical résumés, opposite engagement). A pure
keyword or embedding matcher walks straight into all four traps — the
organizers' own sample ranking scores 0.041 on our benchmark.

**The official scoring.** composite = 0.50·NDCG@10 + 0.30·NDCG@50 +
0.15·MAP + 0.05·P@10, against hidden ground truth, revealed only after
submissions close. No leaderboard, 3 submissions max.

**The five evaluation stages** (every claim in the deck maps to defending
one of these):
1. Format validation (auto) — our CSV passes the official validator.
2. Scoring on hidden ground truth.
3. Code reproduction in a sandboxed Docker container (5 min, CPU, 16 GB,
   no network) + honeypot-rate check. Ours: byte-identical, ~50 s, 0
   honeypots.
4. Manual review — reasoning quality (6 checks), methodology coherence,
   git-history authenticity. Ours: all 6 checks pass on all 100 rows;
   20+ commits; experiments log.
5. Defend-your-work interview — must explain and defend the architecture.
   `docs/design_decisions.md` pre-answers the hard questions.

**Our one-line thesis.** Rank the way a great recruiter reads: verify
claims instead of matching keywords, judge the work in the career history
rather than the self-reported skill list, and weigh whether the person can
actually be hired.

**Headline results (our internal benchmark):** composite 0.9880
(worst-case 0.9874 across six label hypotheses); NDCG@10 1.000, P@10 1.000,
MAP 1.000, NDCG@50 0.960; 0 honeypots in top-100; top-100 is 28 tier-5 +
72 tier-4.

**Honest framing rule (applies to the whole deck).** We have no access to
the official ground truth. Every score we quote is on a benchmark we built
ourselves by analysing the pool's structure. The deck leads with
robustness evidence (minimax, perturbation), not the raw 0.988, and never
implies 0.988 is our expected official score.

---

## Slide 1 — Title
**"Ranking like a recruiter, not a keyword matcher."**
Redrob Intelligent Candidate Discovery & Ranking Challenge — Pratham Modi (solo).
Sub: *100,000 candidates → 100, in 14 seconds, on a laptop CPU, with zero API calls.*

*Speaker note:* the one-liner frames the whole submission — every later
slide is evidence for "like a recruiter" and for "14 s / no API calls."

## Slide 2 — The problem is adversarial, not just big
*The challenge's own brief warns of planted traps (quote the bundle):*
- **Keyword-stuffed profiles** — perfect to any lexical matcher; skills with no corroborating history.
- **~80 honeypots** — subtly impossible profiles; >10% in top-100 = disqualification *(count & rule per official spec)*.
- **Plain-language elites** — strong candidates describing real work without buzzwords; invisible to keyword search.
- **Behavioral twins** — identical on paper; one replies to recruiters, the other dormant for months.

Constraints: ≤5 min · CPU only · 16 GB · no network · no LLM calls at rank time.
Bottom line: *No leaderboard exists, so we built our own benchmark on the official metric. On it, the organizers' keyword-driven sample scores 0.041; our pipeline scores 0.988.*
**Footnote:** *All scores in this deck are on our internal benchmark (Slide 8); official scores are hidden until close.*

*Speaker note:* the traps are quoted from the brief, not our discovery —
say so. The honeypot count "~80" is theirs; what we know is our checks find
zero in our top-100.

## Slide 3 — Thesis: do what a great recruiter does
1. **Verify** — a résumé whose dates don't add up is discarded, not scored.
2. **Read the work, not the list** — judge what they did in roles, not self-tagged skills.
3. **Weigh hireability** — a perfect candidate who never replies isn't a candidate.

*Speaker note:* these three map 1:1 to the consistency gate, the
evidence-from-career-text design, and the availability multiplier.

## Slide 4 — Architecture
Funnel: **100,000 → title gate (−68,821) → consistency gate (−482) → 17,391 scored → top 100.**
`score = skill_fit × experience × location × availability × ownership × penalties`
skill_fit = corroborated-evidence + TF-IDF + MiniLM embedding similarity.
*(Visual: reuse funnel + formula from docs/how_it_works.html.)*

*Speaker note:* multiplicative, not additive — so one strong dimension
can't mask a dead one (the unreachable-ghost problem).

## Slide 5 — The trust layer (every trap dies, no hardcoding)
Four consistency checks: claimed-years vs career span; expert skills never
used; skill durations exceeding the whole career; summary text
contradicting the experience field.
**Calibration rule:** *rare-and-concentrated = planted trap; common-and-uniform = noise.*
(Education gaps fire on 53% of the pool → rejected. Summary contradictions
fire on 0.02% of ordinary profiles but 7.3% of the strongest family →
adopted.)
Result: our checks independently flag **85 suspect profiles** pool-wide
(spec says "~80"), and **zero appear in our top-100**.

*Speaker note:* the 85-vs-80 match is the strongest evidence our checks
find the real planted traps. Never claim we know which are honeypots — we
claim our independent count matches theirs.

## Slide 6 — One real row, end to end
**Our #1 candidate.** Lead AI Engineer at Razorpay, 6.7 yrs, Jaipur.
Résumé: *"rebuilt the candidate-JD matching pipeline, 0.72→0.91 NDCG@10, 50M+ queries/month."*

| Factor | Raw | After weighting | Why |
|---|---|---|---|
| Skill fit | — | **0.955** | Strongest evidence in the pool (evidence 1.00, TF-IDF 1.00, embedding 0.88) |
| Experience | 1.000 | **1.000** | 6.7 yrs — inside the ideal 6–8 band |
| Location | 0.750 | **0.922** | Jaipur — would relocate |
| Availability | 0.814 | **0.898** | 73% reply rate, 30-day notice, active last month |
| Ownership | 1.000 | **1.000** | "owned… designed… serving 50M+" — drove the work |
| Penalties | 1.000 | **1.000** | Nothing the JD rules out |
| **Final** | | **0.7906 → rank 1 of 100,000** | |

The reasoning string is built from these exact factors — real tech, real
signals, honest downside (relocation). Nothing invented.
**Footnote:** *A blind human re-ranking of our top 5 independently picked this same candidate — model and recruiter agree.*

*Speaker note:* full arithmetic — 0.955 × 1.0 × 0.922 × 0.898 × 1.0 × 1.0
= 0.7906. The ownership factor + minimax retune flipped this candidate from
#2 to #1 on its own, matching the human judgment we'd made earlier by hand.

## Slide 7 — Reasoning that survives the audit
The spec's six manual checks (specific facts, JD connection, honest
concerns, no hallucination, variation, rank consistency) — we implemented
all six as code (`audit_reasoning.py`) and run them on **all 100 rows,
every number verified against the profile JSON**. All pass. Concern density
rises with rank (0.68 top half → 0.82 bottom half) — better candidates
genuinely have fewer issues.

*Speaker note:* most teams generate reasoning with an LLM and never check a
row. We test the exact thing the humans grade — and it caught 3
near-duplicate rows during development, which we then fixed.

## Slide 8 — Evaluation without a leaderboard (biggest differentiator)
No feedback, 3 submissions max → we built the missing leaderboard: offline
labels from structural analysis of the pool + the official composite,
implemented exactly. Then we attacked it:
- **Bounded weight search** — refused a +0.004 gain that would have deleted the availability factor (overfitting).
- **Minimax over 6 label hypotheses** we can't distinguish — worst case **0.9874** vs base **0.9880**.
- **±20% weight perturbation** — top-100 stays **97.8%** identical, composite never leaves **0.975–0.991**.
*The number we trust isn't 0.988 — it's the 0.001 gap between best and worst case.*

*Speaker note:* this is the answer to "how do you know you didn't overfit
your own labels?" — minimax + perturbation. Lead with it.

## Slide 9 — Experiments: tried, measured, rejected
- Ablation ladder: features-only **0.700** → +TF-IDF **0.777** → +embeddings+tuning **0.982** → +ownership **0.988**.
- Weight-flip (embedding-dominant): drops to **0.842**, 52 generic SWEs leak in → lexical signal is the engine, embeddings the rescue.
- Gate audit: **0/68,821** false negatives, checked exhaustively.
- Behavioral twins separate as designed (rank 79 vs 100; 93 vs excluded).
- Rejected with evidence: LLM re-ranking (banned at rank time), BGE-large (errors aren't embedding-bound).

*Speaker note:* the weight-flip and BGE results pre-empt two obvious
"why didn't you just…" questions.

## Slide 10 — Constraints: proven, not promised
Reproduced from a fresh clone in Docker (`python:3.11-slim`,
`--network none`, `-m 16g`): **byte-identical CSV, ~50 s wall, <1 GB peak**
(limits 300 s / 16 GB). One command — `make judge` — runs rank → official
validator → full reasoning audit. Artifacts via GitHub release + rebuild
script. 20+ commits of real iteration with an experiments log.

*Speaker note:* this is the Stage-3 answer. "Reproduced, not promised" —
we ran it under their exact constraints ourselves.

## Slide 11 — Limitations, stated plainly
Our benchmark is a *hypothesis* about hidden labels (mitigated by minimax +
perturbation, not eliminated). Concept lexicons are per-JD configuration,
not learned. Availability half-life (45 days) is a product assumption a
live system should fit to data. *We'd rather name these than have you find them.*

*Speaker note:* honesty about limitations, backed by numbers, is the
rarest thing in a hackathon final — it flips the Stage-5 dynamic.

## Slide 12 — Closing: we executed your 90-day plan
Your JD: weeks 1–3 audit → weeks 4–8 ship a v2 ranker → weeks 9–12 build
eval infrastructure. This repo *is* that plan: forensic audit of the pool →
hybrid ranker (14 s, CPU, offline) → benchmark + minimax tuning + automated
audit. *"We'd rather see 10 great matches than 1000 maybes."* Our top-10 is
exactly that — and we can defend every row.

*Speaker note:* the interview is with the team that wrote that JD. This
reframes you from "contestant who ranked candidates" to "candidate who
already did the job."

---

## Design notes
- Dark theme matching `docs/how_it_works.html`, which already has the
  funnel and factor-table visuals to reuse on Slides 4 and 6.
- One idea per slide; big numbers; no stock imagery.
- Optional appendix: full experiments table (experiments.md) + an
  `audit_reasoning.py` output screenshot.

## Number-provenance map (re-verify before export if anything is retuned)
- Funnel counts, rank-1 row, factor breakdown → `submission.csv`,
  `artifacts/weights.json`, run `rank.py`.
- Composite / NDCG / MAP / P@10 / honeypots → `python eval_harness.py submission.csv`.
- Minimax worst-case 0.9874 → `python tune_minimax.py`.
- Perturbation 97.8% / 0.975–0.991 → experiment G in `experiments.md`.
- Ablation ladder, weight-flip, gate audit, twins → `experiments.md`.
- Docker timing / byte-identical → experiment F in `experiments.md`.
