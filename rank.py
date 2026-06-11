"""
Candidate ranking pipeline. Produces the top-100 submission CSV.

Usage:
  python rank.py --candidates path/to/candidates.jsonl \
                 --jd data/job_description.md \
                 --out submission.csv

Pipeline, in order:
  1. Parse the JD into a structured spec (experience band, location policy,
     competency groups, penalty patterns).
  2. Stream candidates. Apply two hard gates:
       a. relevance gate: no software, data or ML title anywhere in the
          career history means the profile cannot fit a hands-on AI role
       b. consistency gate: profiles whose dates and durations are
          internally impossible are excluded
  3. Score the survivors:
       skill_fit  = blend of concept-coverage evidence, TF-IDF similarity
                    and embedding similarity between the profile's career
                    evidence text and the JD
       final      = skill_fit
                    * yoe_score        ^ w_yoe
                    * location_score   ^ w_loc
                    * availability     ^ w_avail
                    * penalty_factor   ^ w_pen
     Exponent weights live in artifacts/weights.json (produced by
     tune_weights.py); sensible defaults are used when the file is absent.
  4. Take the top 100, generate per-candidate reasoning directly from the
     scoring factors (specific facts in, honest concerns included), and
     write the CSV with rank-consistent, deterministically tie-broken
     scores.

The ranking step uses no network access and runs on CPU well inside the
five-minute budget; the only heavyweight work (embedding 100K profiles)
happens once in precompute.py.
"""
import argparse
import csv
import json
import math
import os
import pickle
import sys
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from jd_parser import parse_jd, jd_query_text, CONCEPT_LEXICONS  # noqa: E402
from consistency import consistency_violations  # noqa: E402
from features import (candidate_evidence_text, evidence_score, yoe_score,  # noqa: E402
                      location_score, availability_score, penalty_factors,
                      engineering_title_gate)

ART = os.path.join(HERE, "artifacts")

DEFAULT_WEIGHTS = {
    "w_evidence": 0.5,   # blend weight of concept coverage inside skill_fit
    "w_tfidf": 0.2,      # blend weight of TF-IDF similarity
    "w_embed": 0.3,      # blend weight of embedding similarity
    "w_yoe": 1.0,
    "w_loc": 1.0,
    "w_avail": 1.0,
    "w_pen": 1.0,
}


def load_weights(path):
    weights = dict(DEFAULT_WEIGHTS)
    if path and os.path.exists(path):
        with open(path) as f:
            weights.update(json.load(f))
    return weights


def load_relevance_artifacts(jd_text_str):
    """Returns (ids_order, tfidf_scores, embed_scores) aligned by candidate
    index, or Nones when artifacts are unavailable (pure-feature fallback)."""
    tfidf_path = os.path.join(ART, "tfidf.pkl")
    emb_path = os.path.join(ART, "embeddings.npz")
    if not (os.path.exists(tfidf_path) and os.path.exists(emb_path)):
        return None, None, None

    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    data = np.load(emb_path, allow_pickle=False)
    ids = [i for i in data["ids"]]
    emb = data["embeddings"]

    jd_emb = None
    try:  # local model cache; no network in the ranking environment
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu",
                                    local_files_only=True)
        jd_emb = model.encode([jd_text_str], normalize_embeddings=True,
                              convert_to_numpy=True)[0].astype(np.float32)
    except Exception:
        jd_path = os.path.join(ART, "jd_embedding.npy")
        if os.path.exists(jd_path):
            jd_emb = np.load(jd_path)

    embed_scores = emb @ jd_emb if jd_emb is not None else None
    return ids, tfidf, embed_scores


def minmax(values):
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def build_reasoning(entry, spec, rank):
    """Reasoning assembled from the candidate's own scoring factors. Every
    statement corresponds to a profile fact; concerns are stated when the
    factors found any."""
    c = entry["candidate"]
    p = c["profile"]
    rs = c["redrob_signals"]

    evidence_bits = entry["evidence_concepts"]
    concept_names = {
        "embeddings_retrieval": "production embeddings/retrieval work",
        "vector_search_infra": "vector search infrastructure experience",
        "ranking_systems": "shipped ranking/recommendation systems",
        "ranking_evaluation": "rigorous ranking evaluation experience",
        "python_engineering": "strong Python/ML engineering",
        "llm_finetuning": "LLM fine-tuning exposure",
        "ltr_models": "learning-to-rank modeling",
        "marketplace_domain": "marketplace/HR-tech domain background",
        "scale_inference": "large-scale systems experience",
    }
    strengths = [concept_names[k] for k in evidence_bits if k in concept_names][:3]

    # opener varies with profile facts, not with a random seed
    opener = (f"{p['current_title']} at {p['current_company']} "
              f"({p['years_of_experience']:.1f} yrs)")

    mid = "; ".join(strengths) if strengths else "adjacent technical background"

    engagement_facts = []
    inactive_days = (date(2026, 6, 11) - date(*map(int, rs["last_active_date"].split("-")))).days
    if inactive_days <= 45 and rs["recruiter_response_rate"] >= 0.5:
        engagement_facts.append(
            f"responsive on platform ({rs['recruiter_response_rate']:.0%} reply rate, "
            f"active {rs['last_active_date']})")
    elif rs["open_to_work_flag"]:
        engagement_facts.append(f"open to work, {rs['notice_period_days']}-day notice")
    else:
        engagement_facts.append(
            f"{rs['recruiter_response_rate']:.0%} recruiter reply rate")

    concerns = list(entry["penalty_reasons"])
    yoe = p["years_of_experience"]
    if yoe < spec.yoe_min:
        concerns.append(f"{yoe:.1f} yrs is under the {spec.yoe_min:.0f}-{spec.yoe_max:.0f} yr band")
    elif yoe > spec.yoe_max:
        concerns.append(f"{yoe:.1f} yrs exceeds the {spec.yoe_min:.0f}-{spec.yoe_max:.0f} yr band")
    if p["country"] != spec.country:
        concerns.append(f"based in {p['location']}, {p['country']} and the role does not sponsor visas")
    if rs["notice_period_days"] > spec.notice_pref_days + spec.notice_buyout_days:
        concerns.append(f"{rs['notice_period_days']}-day notice period")
    if inactive_days > 90:
        concerns.append(f"inactive for roughly {inactive_days // 30} months")

    text = f"{opener}: {mid}; {engagement_facts[0]}"
    if concerns:
        text += ". Concerns: " + "; ".join(concerns[:2])
    return text + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd", default=os.path.join(HERE, "data", "job_description.md"))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--weights", default=os.path.join(ART, "weights.json"))
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    today = date(2026, 6, 11)
    weights = load_weights(args.weights)
    spec = parse_jd(open(args.jd).read())
    query = jd_query_text(spec)

    ids_order, tfidf, embed_scores = load_relevance_artifacts(query)
    id_to_idx = {cid: i for i, cid in enumerate(ids_order)} if ids_order else {}

    survivors = []
    evidence_texts = []
    n_total = n_gated_title = n_gated_consistency = 0

    with open(args.candidates) as f:
        for line in f:
            c = json.loads(line)
            n_total += 1
            if not engineering_title_gate(c):
                n_gated_title += 1
                continue
            if consistency_violations(c, today):
                n_gated_consistency += 1
                continue
            text = candidate_evidence_text(c)
            ev = evidence_score(c, spec, text)
            if ev <= 0.15:  # no demonstrated overlap with any must-have
                continue
            hit_concepts = [
                k for k in (spec.must_have + spec.nice_to_have)
                if any(form in text for form in CONCEPT_LEXICONS.get(k, []))
            ]
            pen, pen_reasons = penalty_factors(c, text)
            survivors.append({
                "candidate": c,
                "evidence": ev,
                "evidence_concepts": hit_concepts,
                "yoe": yoe_score(c, spec),
                "loc": location_score(c, spec),
                "avail": availability_score(c, spec, today),
                "pen": pen,
                "penalty_reasons": pen_reasons,
            })
            evidence_texts.append(text)

    print(f"candidates: {n_total}; gated by title: {n_gated_title}; "
          f"gated by consistency: {n_gated_consistency}; "
          f"scored: {len(survivors)}")

    # hybrid relevance over survivors only
    ev_norm = minmax([s["evidence"] for s in survivors])
    if tfidf is not None:
        qv = tfidf.transform([query])
        cv = tfidf.transform(evidence_texts)
        tf_norm = minmax((cv @ qv.T).toarray().ravel())
    else:
        tf_norm = np.zeros(len(survivors), dtype=np.float32)
    if embed_scores is not None:
        es = [embed_scores[id_to_idx[s["candidate"]["candidate_id"]]]
              for s in survivors]
        em_norm = minmax(es)
    else:
        em_norm = np.zeros(len(survivors), dtype=np.float32)

    blend_total = weights["w_evidence"] + weights["w_tfidf"] + weights["w_embed"]
    for i, s in enumerate(survivors):
        skill_fit = (weights["w_evidence"] * ev_norm[i]
                     + weights["w_tfidf"] * tf_norm[i]
                     + weights["w_embed"] * em_norm[i]) / blend_total
        s["skill_fit"] = float(skill_fit)
        s["final"] = float(
            skill_fit
            * s["yoe"] ** weights["w_yoe"]
            * s["loc"] ** weights["w_loc"]
            * s["avail"] ** weights["w_avail"]
            * s["pen"] ** weights["w_pen"]
        )

    # round to emitted precision before sorting so equal emitted scores are
    # tie-broken by candidate_id ascending, as the validator requires
    for s in survivors:
        s["emit_score"] = round(s["final"], 4)
    survivors.sort(key=lambda s: (-s["emit_score"], s["candidate"]["candidate_id"]))
    top = survivors[:args.top]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, s in enumerate(top, 1):
            w.writerow([
                s["candidate"]["candidate_id"], i, f"{s['emit_score']:.4f}",
                build_reasoning(s, spec, i),
            ])
    print(f"wrote top-{len(top)} -> {args.out}")


if __name__ == "__main__":
    main()
