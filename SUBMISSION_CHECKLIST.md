# Submission Checklist

Tracks every required item from submission_spec.md Section 10. Update as
items close.

## 1. The CSV (spec section 2-3)

- [x] Top-100 ranking produced by `python rank.py ...` (submission.csv)
- [x] Passes the official validator (validate_submission.py)
- [x] Reasoning column passes all six manual-review checks on every row
      (audit_reasoning.py)
- [x] Renamed to **team_Hack_IT_On.csv** (validated after renaming;
      composite 0.988, 0 honeypots). Kept as .csv — the spec auto-rejects
      .xlsx/.json submissions (Section 6).

## 2. Portal metadata (entered at upload time)

- [x] Team name (registered)            (Hack_IT_On)
- [x] Primary contact name              (Pratham Modi)
- [x] Primary contact email             (pratham@a79.ai)
- [x] Primary contact phone             (+91-7206680039)
- [x] GitHub repository URL             https://github.com/pratham0039/candidate-ranking
      (grant organizer access at Stage 3; repo is private)
- [x] Sandbox / demo link               (deployed on Streamlit Cloud)
- [x] AI tools declared                 (Claude, Cursor, ChatGPT, Gemini)
- [x] Compute environment summary       (in submission_metadata.yaml)
- [x] Team member list                  (solo)
- [x] Methodology summary, 158/200 words (in submission_metadata.yaml)

## 3. Code repository

- [x] README with setup + single reproduce command
- [x] Full source that produced the CSV, no hidden steps
- [x] Precomputed artifacts: GitHub release artifacts-v1 (embeddings,
      TF-IDF, JD embedding) plus precompute.py to rebuild from scratch
- [x] requirements.txt
- [x] submission_metadata.yaml at repo root (mirror portal values when
      filling the TODOs)
- [x] Dockerfile mirroring evaluation constraints; reproduction verified
      (byte-identical CSV, ~50 s, <1 GB, network off)
- [x] Git history shows real iteration (17 commits, experiments log)

## 4. Sandbox (spec section 10.5) - REQUIRED

- [x] Deployed on Streamlit Community Cloud (sandbox/app.py).
      URL in submission_metadata.yaml. Loader accepts both JSON-array
      (sample_candidates.json) and JSONL uploads; verified end-to-end
      with the 50-row sample.

## 5. The deck (required)

- [x] Mandatory Redrob template filled (11 slides, charts embedded) and
      exported to PDF. Named team_Hack_IT_On.pdf for upload.

## 6. Final pre-upload ritual

1. [x] `make judge` - all green (validator + reasoning audit)
2. [x] `eval_harness.py team_Hack_IT_On.csv` - composite 0.988, 0 honeypots
3. [x] CSV renamed to team_Hack_IT_On.csv; validated after renaming
4. [x] submission_metadata.yaml complete (team, phone, repo, sandbox)
5. [ ] Confirm GitHub release assets download (logged-out browser)
6. [ ] Confirm sandbox link loads end-to-end (incognito window)
7. [ ] Upload on the portal: team_Hack_IT_On.csv + team_Hack_IT_On.pdf +
      metadata; save the confirmation
8. [ ] Do not resubmit without a measured improvement: ties break by
      earlier timestamp, and the last submission is the one that counts

## Upload bundle (final filenames)

- team_Hack_IT_On.csv   - the ranking (CSV, NOT xlsx - spec rejects xlsx)
- team_Hack_IT_On.pdf   - the deck
- GitHub repo: https://github.com/pratham0039/candidate-ranking (grant
  organizer access at Stage 3; repo is private)
- Sandbox URL + portal metadata from submission_metadata.yaml

## Open decisions

- [ ] Whether to commit the offline benchmark (ground_truth.py,
      eval_harness.py, make_gold100.py) before final submission -
      recommended for Stage 4 verifiability; currently kept local
