# Design Decisions and Known Limitations

The hard questions about this system, asked by ourselves first, with the
evidence for each answer. Numbers reference experiments.md.

## "Your benchmark says 0.988. How do you know that number means anything about the hidden ground truth?"

We don't claim it does, point for point. The offline labels are our
hypothesis about the hidden tiers, and the consistency gate shares
assumptions with how those labels were built, so the headline number is
partly self-referential. What we claim, with evidence, is robustness:

- Weights were tuned with bounded ranges so factors the labels cannot
  observe (availability, location) could not be tuned away (experiment A).
- Under +-20% perturbation of every weight, the top-100 stays 97.8%
  identical and the composite never leaves 0.975-0.991 (experiment G).
- We minimaxed over six label hypotheses we could not distinguish
  (engagement baked into tiers or not, the middle family promoted or
  demoted, ambiguous flags planted or noise, location baked in or not).
  The adopted weights have worst-case composite 0.987 across all six
  (tune_minimax.py). The ranking is driven by the factor design, not by a
  lucky labeling.

## "Why is the research-only disqualifier in the JD not implemented?"

Calibration before implementation (our rule from experiment C). We
measured prevalence across all 100,000 profiles: zero research-only
careers, zero LangChain mentions, zero architect-titled roles without
hands-on language (experiment H). All three checks would be dead code on
this pool. The lexicons for them exist in jd_parser.py; wiring them in
costs one line each if the pool changes.

## "Your title gate removes 69% of the pool. How do you know it has no false negatives?"

We audited every rejected profile, not a sample: zero of 68,821 rejected
profiles contain strong retrieval/ML evidence in their career descriptions
(experiment C). The gate is also deliberately the weakest link we allow:
it only excludes profiles with no software, data, or ML title in their
entire career history, which for a hands-on senior AI role is a
requirement a human recruiter applies in the first second.

## "Couldn't your ownership markers just be fingerprinting this synthetic dataset?"

The marker lists are generic recruiting heuristics (owned/designed/shipped
vs lightweight/side-project/handled-by-another-team), not strings unique
to this dataset. The factor is bounded (0.3 to 1.0), tuned rather than
hardcoded, and experiment D shows the failure mode it fixes is a real
ranking failure (strong-but-quiet profiles losing to weak-but-eager ones),
not a label artifact: the same inversion is visible by reading the
profiles directly.

## "Why MiniLM and not a stronger embedding model?"

Measured, not assumed: embedding-dominant blends made results worse
(experiment B), because the binding errors are about evidence
verification, not embedding resolution. NDCG@10 is saturated at MiniLM
with 4x less precompute than BGE-large. If the candidate pool were real
prose rather than templated text, we would revisit this and expect a
larger model to matter more.

## "Why multiplicative scoring instead of a learned model?"

Three reasons. The product form encodes a real hiring rule: no strength
compensates for an absent dimension (an unreachable candidate is not
hireable at any skill level). It is interpretable enough to generate
honest per-candidate reasoning from the factors themselves, which a
learned blackbox would not give us. And with ~100 effective labels worth
of supervision, a learned model would memorize the families, not the task.

## "What would break first in production?"

The concept lexicons. They are seeded from this JD and expanded with
standard synonyms; a genuinely new role family (say, embedded firmware)
needs a new lexicon block, which is per-JD configuration, not engine code.
Second: the availability half-life (45 days) is a product decision frozen
as a constant; a real deployment should fit it to observed
reply-conversion data. Third: the reference date is frozen for
reproducibility and must be unfrozen for a live system.
