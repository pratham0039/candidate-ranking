# Convenience targets. CANDIDATES must point at candidates.jsonl.
CANDIDATES ?= ./candidates.jsonl
PY ?= python

rank:
	$(PY) rank.py --candidates $(CANDIDATES) --jd data/job_description.md --out submission.csv

audit:
	$(PY) audit_reasoning.py --candidates $(CANDIDATES) --submission submission.csv

# Everything a grader needs in one command: produce the CSV, check the
# format, and audit every reasoning row against the profile data.
judge: rank
	$(PY) validate_submission.py submission.csv || true
	$(PY) audit_reasoning.py --candidates $(CANDIDATES) --submission submission.csv

.PHONY: rank audit judge
