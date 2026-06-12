"""
Weight tuning for the ranking formula.

The final score multiplies interpretable factors (experience fit, location,
availability, penalties) with a blended skill-fit relevance score. The
relative influence of each factor is a free parameter. This script searches
those parameters to maximize the competition's composite metric against an
offline labels file, built and maintained separately as our internal
benchmark (labels are not distributed with the repo).

    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

The candidate pipeline (gates, factor extraction, relevance) runs once;
each weight sample only recombines precomputed factors, so thousands of
samples take seconds.

Usage:
  python tune_weights.py --candidates path/to/candidates.jsonl \
                         --labels artifacts/labels.csv \
                         --samples 2000
Writes the best weights to artifacts/weights.json.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from jd_parser import parse_jd, jd_query_text, CONCEPT_LEXICONS  # noqa: E402
from consistency import consistency_violations  # noqa: E402
from features import (candidate_evidence_text, evidence_score, yoe_score,  # noqa: E402
                      location_score, availability_score, penalty_factors,
                      engineering_title_gate, ownership_score)
from rank import load_relevance_artifacts, minmax  # noqa: E402
from constants import REFERENCE_DATE  # noqa: E402

ART = os.path.join(HERE, "artifacts")


def load_labels(path):
    labels = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            labels[row["candidate_id"]] = int(row["tier"])
    return labels


def composite_metric(ranked_ids, labels, total_relevant, ideal_gains_10,
                     ideal_gains_50):
    tiers = [labels.get(cid, 0) for cid in ranked_ids]
    gains = [2 ** t - 1 for t in tiers]

    def dcg(gs):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gs))

    ndcg10 = dcg(gains[:10]) / dcg(ideal_gains_10)
    ndcg50 = dcg(gains[:50]) / dcg(ideal_gains_50)
    rel = [1 if t >= 3 else 0 for t in tiers]
    p10 = sum(rel[:10]) / 10
    hits, ap_sum = 0, 0.0
    for i, r in enumerate(rel):
        if r:
            hits += 1
            ap_sum += hits / (i + 1)
    ap = ap_sum / min(total_relevant, len(rel))
    return 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * ap + 0.05 * p10


def collect_factors(candidates_path, spec, today):
    query = jd_query_text(spec)
    ids_order, tfidf, embed_scores = load_relevance_artifacts(query)
    id_to_idx = {cid: i for i, cid in enumerate(ids_order)} if ids_order else {}

    rows, texts = [], []
    with open(candidates_path) as f:
        for line in f:
            c = json.loads(line)
            if not engineering_title_gate(c):
                continue
            if consistency_violations(c, today):
                continue
            text = candidate_evidence_text(c)
            ev = evidence_score(c, spec, text)
            if ev <= 0.15:
                continue
            pen, _ = penalty_factors(c, text)
            rows.append({
                "id": c["candidate_id"],
                "evidence": ev,
                "yoe": yoe_score(c, spec),
                "loc": location_score(c, spec),
                "avail": availability_score(c, spec, today),
                "pen": pen,
                "own": ownership_score(c, text),
            })
            texts.append(text)

    ev_norm = minmax([r["evidence"] for r in rows])
    if tfidf is not None:
        qv = tfidf.transform([query])
        tf_norm = minmax((tfidf.transform(texts) @ qv.T).toarray().ravel())
    else:
        tf_norm = np.zeros(len(rows), dtype=np.float32)
    if embed_scores is not None:
        em_norm = minmax([embed_scores[id_to_idx[r["id"]]] for r in rows])
    else:
        em_norm = np.zeros(len(rows), dtype=np.float32)

    ids = [r["id"] for r in rows]
    factors = {
        "ev": ev_norm.astype(np.float64),
        "tf": tf_norm.astype(np.float64),
        "em": em_norm.astype(np.float64),
        "yoe": np.array([r["yoe"] for r in rows]),
        "loc": np.array([r["loc"] for r in rows]),
        "avail": np.array([r["avail"] for r in rows]),
        "pen": np.array([r["pen"] for r in rows]),
        "own": np.array([r["own"] for r in rows]),
    }
    return ids, factors


def score_with(weights, factors):
    blend_total = weights["w_evidence"] + weights["w_tfidf"] + weights["w_embed"]
    skill = (weights["w_evidence"] * factors["ev"]
             + weights["w_tfidf"] * factors["tf"]
             + weights["w_embed"] * factors["em"]) / blend_total
    return (skill
            * factors["yoe"] ** weights["w_yoe"]
            * factors["loc"] ** weights["w_loc"]
            * factors["avail"] ** weights["w_avail"]
            * factors["pen"] ** weights["w_pen"]
            * factors["own"] ** weights.get("w_own", 1.0))


def sample_weights(rng):
    """Bounded search ranges. Lower bounds keep the factors the JD
    explicitly asks for (availability, location, penalties) from being
    tuned away entirely; the offline labels cannot fully observe their
    effect, so removing them would overfit the benchmark."""
    return {
        "w_evidence": rng.uniform(0.1, 1.0),
        "w_tfidf": rng.uniform(0.1, 1.0),
        "w_embed": rng.uniform(0.1, 1.0),
        "w_yoe": rng.uniform(0.3, 2.0),
        "w_loc": rng.uniform(0.2, 1.5),
        "w_avail": rng.uniform(0.3, 1.5),
        "w_pen": rng.uniform(0.5, 2.5),
        "w_own": rng.uniform(0.3, 3.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", default=os.path.join(ART, "labels.csv"))
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    today = REFERENCE_DATE
    spec = parse_jd(open(os.path.join(HERE, "data", "job_description.md")).read())
    labels = load_labels(args.labels)

    all_tiers = sorted(labels.values(), reverse=True)
    ideal10 = [2 ** t - 1 for t in all_tiers[:10]]
    ideal50 = [2 ** t - 1 for t in all_tiers[:50]]
    total_relevant = sum(1 for t in labels.values() if t >= 3)

    print("Collecting factors (single pass)...")
    ids, factors = collect_factors(args.candidates, spec, today)
    ids_arr = np.array(ids)
    print(f"{len(ids)} scored candidates")

    rng = random.Random(args.seed)
    from rank import DEFAULT_WEIGHTS
    best_w, best_score = dict(DEFAULT_WEIGHTS), -1.0

    candidates_w = [dict(DEFAULT_WEIGHTS)] + [sample_weights(rng)
                                              for _ in range(args.samples)]
    for i, w in enumerate(candidates_w):
        scores = score_with(w, factors)
        top_idx = np.argsort(-scores)[:args.top]
        comp = composite_metric(list(ids_arr[top_idx]), labels,
                                total_relevant, ideal10, ideal50)
        if comp > best_score:
            best_score = comp
            best_w = w
            print(f"  sample {i}: composite {comp:.4f}  {json.dumps({k: round(v, 2) for k, v in w.items()})}")

    out = os.path.join(ART, "weights.json")
    with open(out, "w") as f:
        json.dump(best_w, f, indent=2)
    print(f"\nBest composite: {best_score:.4f}")
    print(f"Weights written -> {out}")


if __name__ == "__main__":
    main()
