"""
Offline precompute step. Run once before ranking; rank.py loads the
artifacts this produces. This step may take longer than the ranking budget
(it downloads a small embedding model and encodes 100K profiles on CPU),
which is why it is separated from rank.py.

Produces in artifacts/:
  embeddings.npz   float32 matrix [n_candidates, 384] of profile embeddings
                   (all-MiniLM-L6-v2 over career evidence text) plus the
                   candidate id order
  tfidf.pkl        fitted TfidfVectorizer over the same evidence texts

Usage:
  python precompute.py --candidates path/to/candidates.jsonl
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from features import candidate_evidence_text  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd", default=os.path.join(HERE, "data", "job_description.md"))
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    ids, texts = [], []
    with open(args.candidates) as f:
        for line in f:
            c = json.loads(line)
            ids.append(c["candidate_id"])
            texts.append(candidate_evidence_text(c))
    print(f"Loaded {len(ids)} candidates")

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, stop_words="english",
    )
    tfidf.fit(texts)
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    print("TF-IDF vectorizer fitted and saved")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    np.savez_compressed(
        os.path.join(ART, "embeddings.npz"),
        ids=np.array(ids), embeddings=emb,
    )
    print(f"Embeddings saved: {emb.shape}")

    # JD query embedding, used by rank.py when the model cache is absent
    # in the ranking environment
    from jd_parser import parse_jd, jd_query_text
    spec = parse_jd(open(args.jd).read())
    jd_emb = model.encode([jd_query_text(spec)], normalize_embeddings=True,
                          convert_to_numpy=True)[0].astype(np.float32)
    np.save(os.path.join(ART, "jd_embedding.npy"), jd_emb)
    print("JD embedding saved")


if __name__ == "__main__":
    main()
