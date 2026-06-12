"""
Minimax weight tuning over ground-truth hypotheses.

Our offline labels are one hypothesis about the hidden ground truth. Where
the data cannot distinguish between plausible labelings (does engagement
lower a candidate's tier, or only their ideal rank? are the ambiguous
skill-duration profiles planted or noise? how is the "growing into AI"
family valued?), tuning against a single hypothesis risks overfitting to
our guess. This script generates several deliberately different label
variants spanning those open questions and searches for weights that
maximize the MINIMUM composite across all of them: weights that are good
under every hypothesis we cannot rule out, rather than perfect under one.

Variants:
  base        the labels as constructed
  engagement  dormant or unresponsive relevant candidates demoted one tier
              (engagement baked into the hidden tiers)
  t3_demoted  the "growing into AI" family valued at tier 2
  t3_promoted the "growing into AI" family valued at tier 4
  ambig_clean profiles flagged only by the skill-duration check treated as
              clean (that check mistaken, profiles genuine)
  location    elite candidates outside the JD's country demoted one tier
              (hireability baked into the hidden tiers)

Usage:
  python tune_minimax.py --candidates path/to/candidates.jsonl --samples 3000
Writes artifacts/weights_minimax.json and prints a comparison against the
current artifacts/weights.json.
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from jd_parser import parse_jd  # noqa: E402
from constants import REFERENCE_DATE  # noqa: E402
from tune_weights import (collect_factors, score_with, composite_metric,  # noqa: E402
                          sample_weights)

ART = os.path.join(HERE, "artifacts")
GT = os.path.join(ART, "ground_truth.csv")


def build_variants(candidates_path):
    base, families, flags = {}, {}, {}
    with open(GT) as f:
        for row in csv.DictReader(f):
            cid = row["candidate_id"]
            base[cid] = int(row["tier"])
            families[cid] = row["family"]
            flags[cid] = row["flags"]

    # signals needed for the engagement/location variants (relevant ids only)
    relevant = {cid for cid, fam in families.items()
                if fam.startswith(("T3", "T4", "T5"))}
    signals = {}
    with open(candidates_path) as f:
        for line in f:
            c = json.loads(line)
            cid = c["candidate_id"]
            if cid in relevant:
                rs = c["redrob_signals"]
                inactive = (REFERENCE_DATE -
                            __import__("datetime").date(*map(int, rs["last_active_date"].split("-")))).days
                signals[cid] = (rs["recruiter_response_rate"], inactive,
                                c["profile"]["country"])

    variants = {"base": dict(base)}

    v = dict(base)
    for cid in relevant:
        resp, inactive, _ = signals[cid]
        if v.get(cid, 0) >= 3 and (resp < 0.3 or inactive > 90):
            v[cid] -= 1
    variants["engagement"] = v

    v = dict(base)
    for cid, fam in families.items():
        if fam == "T3_ds_growing" and v[cid] == 3:
            v[cid] = 2
    variants["t3_demoted"] = v

    v = dict(base)
    for cid, fam in families.items():
        if fam == "T3_ds_growing" and v[cid] == 3:
            v[cid] = 4
    variants["t3_promoted"] = v

    v = dict(base)
    for cid, fl in flags.items():
        if fl == "skill_duration_exceeds_career" and families[cid] == "T4_mle_prod":
            v[cid] = 4  # treat as clean tier-4 after all
    variants["ambig_clean"] = v

    v = dict(base)
    for cid in relevant:
        _, _, country = signals[cid]
        if v.get(cid, 0) >= 4 and country != "India":
            v[cid] -= 1
    variants["location"] = v

    return variants


def variant_context(labels):
    tiers = sorted(labels.values(), reverse=True)
    return ([2 ** t - 1 for t in tiers[:10]],
            [2 ** t - 1 for t in tiers[:50]],
            sum(1 for t in labels.values() if t >= 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    spec = parse_jd(open(os.path.join(HERE, "data", "job_description.md")).read())
    print("Building label variants...")
    variants = build_variants(args.candidates)
    contexts = {name: variant_context(lbl) for name, lbl in variants.items()}

    print("Collecting factors (single pass)...")
    ids, factors = collect_factors(args.candidates, spec, REFERENCE_DATE)
    ids_arr = np.array(ids)

    def min_composite(w):
        scores = score_with(w, factors)
        top = list(ids_arr[np.argsort(-scores)[:100]])
        per = {}
        for name, lbl in variants.items():
            i10, i50, rel = contexts[name]
            per[name] = composite_metric(top, lbl, rel, i10, i50)
        return min(per.values()), per

    current = json.load(open(os.path.join(ART, "weights.json")))
    cur_min, cur_per = min_composite(current)
    print(f"\ncurrent weights: min composite {cur_min:.4f}")
    for name, v in sorted(cur_per.items()):
        print(f"  {name:12s} {v:.4f}")

    rng = random.Random(args.seed)
    best_w, best_min, best_per = dict(current), cur_min, cur_per
    for i in range(args.samples):
        w = sample_weights(rng)
        m, per = min_composite(w)
        if m > best_min:
            best_min, best_w, best_per = m, w, per
            print(f"  sample {i}: min composite {m:.4f}")

    out = os.path.join(ART, "weights_minimax.json")
    with open(out, "w") as f:
        json.dump(best_w, f, indent=2)
    print(f"\nbest minimax weights: min composite {best_min:.4f} -> {out}")
    for name, v in sorted(best_per.items()):
        print(f"  {name:12s} {v:.4f}")


if __name__ == "__main__":
    main()
