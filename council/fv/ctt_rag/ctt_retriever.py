import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def _tokenize(q: str):
    import re
    return re.findall(r"[a-zA-Z0-9']+", q.lower())

class CTTRetriever:
    def __init__(self, index_dir: str, embed_model: str = "BAAI/bge-small-en-v1.5"):
        import os
        from pathlib import Path
        idx = Path(index_dir)
        self.meta = pickle.loads((idx / "meta.pkl").read_bytes())
        self.bm25 = pickle.loads((idx / "bm25.pkl").read_bytes())
        self.index = faiss.read_index(str(idx / "ctt.faiss"))
        self._device = os.getenv("EMBED_DEVICE", "cpu")
        self.model = SentenceTransformer(embed_model, device=self._device)
        self._reranker = None  # Fix 4: lazy-loaded only when rerank=True

    def retrieve(self, query: str, bm25_k=25, vec_k=25, final_k=8, rerank=False):
        q_tok = _tokenize(query)

        bm25_scores = self.bm25.get_scores(q_tok)
        bm25_top = np.argsort(bm25_scores)[::-1][:bm25_k]

        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        vec_scores, vec_ids = self.index.search(q_emb, vec_k)
        vec_ids, vec_scores = vec_ids[0], vec_scores[0]

        cand = {}
        for i in bm25_top:
            cand[int(i)] = cand.get(int(i), 0.0) + float(bm25_scores[i])
        for i, s in zip(vec_ids, vec_scores):
            if int(i) < 0:
                continue
            cand[int(i)] = cand.get(int(i), 0.0) + float(s) * 10.0

        ranked = sorted(cand.items(), key=lambda x: x[1], reverse=True)
        top_ids = [i for i, _ in ranked[:max(final_k, 20)]]

        if rerank:                                                         # Fix 4
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("BAAI/bge-reranker-base", device=self._device)
                print("[CTT] Reranker loaded (first use)")
            pairs = [(query, self.meta[i]["contextual_text"]) for i in top_ids]
            scores = self._reranker.predict(pairs)
            top_ids = [i for i, _ in sorted(zip(top_ids, scores), key=lambda x: x[1], reverse=True)[:final_k]]
        else:
            top_ids = top_ids[:final_k]

        out = []
        for i in top_ids:
            m = self.meta[i]
            out.append({
                "title": m.get("title"),
                "episode_id": m.get("episode_id"),
                "source_path": m.get("source_path"),
                "chunk_index": m.get("chunk_index"),
                "text": m.get("text"),
            })
        return out
