"""
Hosted sandbox demo for the candidate ranker.

Accepts a small candidate sample (JSONL upload, up to a few hundred rows)
plus an optional job description, runs the full ranking pipeline, and
returns the ranked CSV. Falls back to feature-only scoring when the
precomputed embedding artifacts for the full pool are not present, so the
demo runs end-to-end on any small sample within the compute budget.

Run locally:
  streamlit run sandbox/app.py
"""
import csv
import io
import json
import os
import sys
from datetime import date

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from jd_parser import parse_jd, jd_query_text, CONCEPT_LEXICONS  # noqa: E402
from consistency import consistency_violations  # noqa: E402
from features import (candidate_evidence_text, evidence_score, yoe_score,  # noqa: E402
                      location_score, availability_score, penalty_factors,
                      engineering_title_gate, ownership_score)
from rank import build_reasoning, load_weights, minmax  # noqa: E402
from constants import REFERENCE_DATE  # noqa: E402

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker — sandbox")
st.caption("Upload a small candidates.jsonl sample; get the ranked shortlist.")

default_jd_path = os.path.join(ROOT, "data", "job_description.md")
jd_text = st.text_area(
    "Job description",
    value=open(default_jd_path).read() if os.path.exists(default_jd_path) else "",
    height=200,
)
uploaded = st.file_uploader("candidates.jsonl sample (max ~500 rows)", type=["jsonl", "json"])
top_n = st.slider("Shortlist size", 5, 100, 20)

if uploaded and jd_text.strip() and st.button("Rank candidates"):
    today = REFERENCE_DATE
    spec = parse_jd(jd_text)
    weights = load_weights(os.path.join(ROOT, "artifacts", "weights.json"))

    candidates = []
    for line in uploaded.getvalue().decode("utf-8").splitlines():
        line = line.strip()
        if line:
            candidates.append(json.loads(line))
    st.write(f"Loaded {len(candidates)} candidates")

    survivors, texts = [], []
    for c in candidates:
        if not engineering_title_gate(c):
            continue
        if consistency_violations(c, today):
            continue
        text = candidate_evidence_text(c)
        ev = evidence_score(c, spec, text)
        if ev <= 0.15:
            continue
        pen, pen_reasons = penalty_factors(c, text)
        hit_concepts = []
        hit_forms = []
        for k in spec.must_have + spec.nice_to_have:
            matched = [f for f in CONCEPT_LEXICONS.get(k, []) if f in text]
            if matched:
                hit_concepts.append(k)
                hit_forms.append(max(matched, key=len))
        survivors.append({
            "candidate": c,
            "evidence": ev,
            "evidence_concepts": hit_concepts,
            "hit_forms": hit_forms,
            "yoe": yoe_score(c, spec),
            "loc": location_score(c, spec),
            "avail": availability_score(c, spec, today),
            "pen": pen,
            "penalty_reasons": pen_reasons,
            "own": ownership_score(c, text),
        })
        texts.append(text)

    if not survivors:
        st.warning("No candidate passed the relevance and consistency gates.")
        st.stop()

    # small-sample scoring: evidence + TF-IDF fitted on the sample itself
    ev_norm = minmax([s["evidence"] for s in survivors])
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        mat = tfidf.fit_transform(texts + [jd_query_text(spec)])
        tf_norm = minmax((mat[:-1] @ mat[-1].T).toarray().ravel())
    except Exception:
        tf_norm = [0.0] * len(survivors)

    blend = weights["w_evidence"] + weights["w_tfidf"]
    for i, s in enumerate(survivors):
        skill = (weights["w_evidence"] * ev_norm[i] + weights["w_tfidf"] * tf_norm[i]) / blend
        s["emit_score"] = round(float(
            skill
            * s["yoe"] ** weights["w_yoe"]
            * s["loc"] ** weights["w_loc"]
            * s["avail"] ** weights["w_avail"]
            * s["pen"] ** weights["w_pen"]
            * s["own"] ** weights.get("w_own", 1.0)
        ), 4)

    survivors.sort(key=lambda s: (-s["emit_score"], s["candidate"]["candidate_id"]))
    top = survivors[:top_n]

    rows = []
    for i, s in enumerate(top, 1):
        rows.append({
            "candidate_id": s["candidate"]["candidate_id"],
            "rank": i,
            "score": s["emit_score"],
            "reasoning": build_reasoning(s, spec, i),
        })
    st.dataframe(rows, use_container_width=True)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader()
    w.writerows(rows)
    st.download_button("Download ranked CSV", buf.getvalue(),
                       file_name="ranked_sample.csv", mime="text/csv")
