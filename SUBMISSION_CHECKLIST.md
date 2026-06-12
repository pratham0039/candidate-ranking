# Submission Checklist

Tracks every required item from submission_spec.md Section 10. Update as
items close.

## 1. The CSV (spec section 2-3)

- [x] Top-100 ranking produced by `python rank.py ...` (submission.csv)
- [x] Passes the official validator (validate_submission.py)
- [x] Reasoning column passes all six manual-review checks on every row
      (audit_reasoning.py)
- [ ] **Rename to the registered participant ID** (e.g. team_xxx.csv) once
      known. Do not edit contents after renaming; re-run the validator on
      the renamed file.

## 2. Portal metadata (entered at upload time)

- [ ] Team name (registered)            <- NEEDED FROM YOU
- [x] Primary contact name              (Pratham Modi)
- [x] Primary contact email             (pratham@a79.ai)
- [ ] Primary contact phone             <- NEEDED FROM YOU
- [x] GitHub repository URL             https://github.com/pratham0039/India_runs
      (grant organizer access at Stage 3; repo is private)
- [ ] Sandbox / demo link               <- NEEDS DEPLOY (see below)
- [x] AI tools declared                 (Claude)
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

## 4. Sandbox (spec section 10.5) - REQUIRED, flagged at Stage 1 if missing

App is ready (sandbox/app.py + sandbox/requirements.txt). To deploy on
Streamlit Community Cloud (free, ~5 minutes, needs your GitHub login):

1. Go to https://share.streamlit.io and sign in with the pratham0039
   GitHub account.
2. New app -> repository pratham0039/India_runs -> branch main ->
   main file path: sandbox/app.py
3. Advanced settings -> Python 3.11. Deploy.
4. Verify: upload sample_candidates.json (50 rows from the bundle),
   click Rank candidates, confirm a ranked table + CSV download appears.
5. Put the app URL into submission_metadata.yaml (sandbox_link) and the
   portal form.

Note: the repo is private; Streamlit Cloud will ask for repo access
during setup, grant it. If the deploy fights the private repo, the
fallback the spec allows is the committed Dockerfile recipe.

## 5. The deck (hackathon listing: PDF, required)

- [ ] Deck explaining approach -> PDF. Not started; planned content maps
      to README sections + docs/how_it_works.html visuals +
      experiments.md results.

## 6. Final pre-upload ritual (do in order, ~10 minutes)

1. `make judge CANDIDATES=path/to/candidates.jsonl` - all green
2. `python eval_harness.py submission.csv` (local) - composite as expected
3. Rename CSV to participant ID; re-run validator on the renamed file
4. Fill remaining TODOs in submission_metadata.yaml; commit + push
5. Confirm GitHub release assets still downloadable (logged out browser)
6. Confirm sandbox link loads end-to-end (incognito window)
7. Upload CSV + metadata on the portal; save the confirmation
8. Do not resubmit afterwards without a measured improvement: ties break
   by earlier timestamp, and the last submission is the one that counts

## Open decisions

- [ ] Scrub the two personal files from remote git history (force push,
      needs you: `! cd .../redrob-ranker && git push --force origin main`)
- [ ] Whether to commit the offline benchmark (ground_truth.py,
      eval_harness.py, make_gold100.py) before final submission -
      recommended for Stage 4 verifiability; currently kept local
