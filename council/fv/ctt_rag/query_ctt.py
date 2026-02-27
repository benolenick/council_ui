#!/usr/bin/env python3
import argparse, json, pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def tokenize(q: str):
    import re
    return re.findall(r"[a-zA-Z0-9']+", q.lower())

def load_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--bm25_k", type=int, default=25)
    ap.add_argument("--vec_k", type=int, default=25)
    ap.add_argument("--final_k", type=int, default=8)
    ap.add_argument("--rerank", action="store_true", help="Optional cross-encoder rerank (slower)")
    args = ap.parse_args()

    idx = Path(args.index_dir)
    chunks_path = idx / "chunks.jsonl"
    bm25_path = idx / "bm25.pkl"
    faiss_path = idx / "ctt.faiss"
    meta_path = idx / "meta.pkl"

    meta = pickle.loads(meta_path.read_bytes())
    bm25 = pickle.loads(bm25_path.read_bytes())
    index = faiss.read_index(str(faiss_path))

    model = SentenceTransformer(args.embed_model)

    q = args.q.strip()
    q_tok = tokenize(q)

    # BM25 candidates
    bm25_scores = bm25.get_scores(q_tok)
    bm25_top = np.argsort(bm25_scores)[::-1][:args.bm25_k]

    # Vector candidates
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    vec_scores, vec_ids = index.search(q_emb, args.vec_k)
    vec_ids = vec_ids[0]
    vec_scores = vec_scores[0]

    # Union candidates with blended score
    cand = {}
    for i in bm25_top:
        cand[int(i)] = cand.get(int(i), 0.0) + float(bm25_scores[i])
    for i, s in zip(vec_ids, vec_scores):
        if int(i) < 0:
            continue
        cand[int(i)] = cand.get(int(i), 0.0) + float(s) * 10.0  # scale cosine-ish scores up a bit

    # Sort by blended
    ranked = sorted(cand.items(), key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in ranked[:max(args.final_k, 20)]]

    # Optional rerank
    if args.rerank:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("BAAI/bge-reranker-base")
            pairs = [(q, meta[i]["contextual_text"]) for i in top_ids]
            r = reranker.predict(pairs)
            reranked = sorted(zip(top_ids, r), key=lambda x: x[1], reverse=True)
            top_ids = [i for i, _ in reranked[:args.final_k]]
        except Exception as e:
            print(f"[warn] rerank failed, using blended rank: {e}")
            top_ids = top_ids[:args.final_k]
    else:
        top_ids = top_ids[:args.final_k]

    # Print results with citations
    print("\n=== TOP HITS ===\n")
    for rank, i in enumerate(top_ids, 1):
        m = meta[i]
        excerpt = m["text"].strip().replace("\n", " ")
        if len(excerpt) > 450:
            excerpt = excerpt[:450] + "..."
        print(f"[{rank}] {m.get('title','(no title)')}  (episode_id={m.get('episode_id')})")
        print(f"    source: {m.get('source_path')}")
        print(f"    chunk:  {m.get('chunk_index')}")
        print(f"    text:   {excerpt}")
        print()

if __name__ == "__main__":
    main()
