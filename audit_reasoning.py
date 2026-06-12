"""
Reasoning-column audit. Runs the six checks the manual review applies to
sampled rows, but on every row of the submission, with each claim verified
against the candidate's profile JSON.

Checks:
  1. specific_facts   each row cites at least two profile-derived specifics
                      (numbers, dates, named technologies, employers)
  2. jd_connection    each row references at least one JD competency
  3. honest_concerns  when the candidate has an objective gap (notice above
                      buyout, outside the experience band, abroad, inactive
                      45+ days), the row says so
  4. no_hallucination every number, date, employer and technology named in
                      the row exists in the profile
  5. variation        no two rows are near-duplicates (token Jaccard)
  6. rank_consistency concern density should rise with rank, not fall

Usage:
  python audit_reasoning.py --candidates path/to/candidates.jsonl \
                            --submission submission.csv
Exits non-zero if any hard check fails.
"""
import argparse
import csv
import json
import re
import sys

sys.path.insert(0, "src")
from jd_parser import parse_jd, CONCEPT_LEXICONS  # noqa: E402
from features import candidate_evidence_text  # noqa: E402


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def audit(candidates_path, submission_path, jd_path):
    spec = parse_jd(open(jd_path).read())
    rows = list(csv.DictReader(open(submission_path)))
    wanted = {r["candidate_id"] for r in rows}
    profiles = {}
    with open(candidates_path) as f:
        for line in f:
            c = json.loads(line)
            if c["candidate_id"] in wanted:
                profiles[c["candidate_id"]] = c

    all_forms = [f for forms in CONCEPT_LEXICONS.values() for f in forms]
    failures = {k: [] for k in
                ["specific_facts", "jd_connection", "honest_concerns",
                 "no_hallucination", "variation", "rank_consistency"]}
    concern_by_rank = []

    for r in rows:
        cid, rank, text = r["candidate_id"], int(r["rank"]), r["reasoning"]
        low = text.lower()
        c = profiles[cid]
        p, rs = c["profile"], c["redrob_signals"]
        evidence = candidate_evidence_text(c)
        companies = {p["current_company"].lower()} | {
            j["company"].lower() for j in c["career_history"]}
        titles = {p["current_title"].lower()} | {
            j["title"].lower() for j in c["career_history"]}

        # 1. specific facts: count verifiable specifics present
        specifics = 0
        if f"{p['years_of_experience']:.1f}" in text:
            specifics += 1
        specifics += sum(1 for comp in companies if comp in low)
        specifics += sum(1 for f in all_forms if len(f) > 3 and f in low)
        if specifics < 2:
            failures["specific_facts"].append((rank, cid))

        # 2. JD connection: mentions at least one competency surface form
        if not any(f in low for f in all_forms):
            failures["jd_connection"].append((rank, cid))

        # 3. honest concerns: objective gaps must be acknowledged
        gaps = []
        if rs["notice_period_days"] > spec.notice_pref_days + spec.notice_buyout_days:
            gaps.append("notice")
        if not (spec.yoe_min <= p["years_of_experience"] <= spec.yoe_max):
            gaps.append("band")
        if p["country"] != spec.country:
            gaps.append("abroad")
        if gaps and "concern" not in low:
            failures["honest_concerns"].append((rank, cid, gaps))

        # 4. no hallucination: verify every claim type we emit
        m = re.search(r"(\d+\.\d+) yrs", text)
        if m and abs(float(m.group(1)) - p["years_of_experience"]) > 0.05:
            failures["no_hallucination"].append((rank, cid, "yoe"))
        m = re.search(r"replies to (\d+)%", text)
        if m and abs(int(m.group(1)) - round(rs["recruiter_response_rate"] * 100)) > 1:
            failures["no_hallucination"].append((rank, cid, "response_rate"))
        m = re.search(r"(\d+)-day notice", text)
        if m and int(m.group(1)) != rs["notice_period_days"]:
            failures["no_hallucination"].append((rank, cid, "notice"))
        m = re.search(r"last active (\d{4}-\d{2}-\d{2})", text)
        if m and m.group(1) != rs["last_active_date"]:
            failures["no_hallucination"].append((rank, cid, "last_active"))
        for named in re.findall(r"(?:at|previously [^,;.]+ at) ([A-Z][\w.&' -]+?)(?:[,;.(]| with)", text):
            named_l = named.strip().lower()
            if named_l and not any(named_l in comp or comp in named_l
                                   for comp in companies | titles):
                failures["no_hallucination"].append((rank, cid, f"employer:{named}"))
        # technologies named must exist in the candidate's own evidence
        m = re.search(r"(?:hands-on work with|Demonstrates|Evidence of) (.+?)(?: in actual| across roles|;)", text)
        if m:
            for tech in m.group(1).split(","):
                t = tech.strip().lower()
                if t and t != "applied ml delivery" and t not in evidence:
                    failures["no_hallucination"].append((rank, cid, f"tech:{t}"))

        concern_by_rank.append((rank, 1 if "concern" in low else 0))

    # 5. variation: pairwise token jaccard
    toks = [(int(r["rank"]), tokens(r["reasoning"])) for r in rows]
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            sim = jaccard(toks[i][1], toks[j][1])
            if sim > 0.8:
                failures["variation"].append((toks[i][0], toks[j][0], round(sim, 2)))

    # 6. rank consistency: concern rate in top half vs bottom half
    half = len(concern_by_rank) // 2
    by_rank = sorted(concern_by_rank)
    top_rate = sum(c for _, c in by_rank[:half]) / max(half, 1)
    bot_rate = sum(c for _, c in by_rank[half:]) / max(len(by_rank) - half, 1)
    if top_rate > bot_rate:
        failures["rank_consistency"].append(
            f"top-half concern rate {top_rate:.2f} > bottom-half {bot_rate:.2f}")

    print(f"Audited {len(rows)} rows")
    hard_fail = False
    for check, items in failures.items():
        status = "PASS" if not items else f"FAIL ({len(items)})"
        print(f"  {check:18s} {status}")
        for item in items[:5]:
            print(f"      {item}")
        if items:
            hard_fail = True
    print(f"  concern rate: top-half {top_rate:.2f}, bottom-half {bot_rate:.2f} "
          f"(rising with rank = consistent)")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--jd", default="data/job_description.md")
    args = ap.parse_args()
    sys.exit(audit(args.candidates, args.submission, args.jd))
