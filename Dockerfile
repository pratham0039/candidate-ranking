# Reproduction environment for the ranking step. Mirrors the evaluation
# constraints: CPU only, 16 GB RAM, no network during ranking.
#
# Build:
#   docker build -t redrob-ranker .
# Run the ranking step exactly as the evaluators would (note --network none):
#   docker run --rm --network none -m 16g \
#     -v /path/to/candidates.jsonl:/work/candidates.jsonl:ro \
#     -v $(pwd)/artifacts:/work/artifacts \
#     redrob-ranker \
#     python rank.py --candidates /work/candidates.jsonl \
#                    --jd data/job_description.md --out /work/artifacts/submission.csv
#
# Measured on this image: ranking 100K candidates completes in ~50 s
# wall-clock with <1 GB peak memory, zero network calls.
FROM python:3.11-slim

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY rank.py precompute.py audit_reasoning.py ./
