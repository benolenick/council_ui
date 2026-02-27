#!/usr/bin/env python3
import os, re, json, argparse, pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi


TIME_LINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}")
TAG_RE = re.compile(r"<[^>]+>")  # removes <c> tags etc.

def clean_vtt_or_text(raw: str) -> str:
    lines = raw.splitlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if TIME_LINE_RE.match(line):
            continue
        # remove cue formatting
        line = TAG_RE.sub("", line)
        line = line.replace("\ufeff", "").strip()
        # drop purely numeric cue ids
        if line.isdigit():
            continue
        # collapse repeated spaces
        line = re.sub(r"\s+", " ", line)
        out.append(line)

    # De-dupe immediate repeats (common in VTT where cue + plain line duplicates)
    deduped = []
    prev = None
    for l in out:
        if l == prev:
            continue
        deduped.append(l)
        prev = l

    text = "\n".join(deduped).strip()

    # Optional: join broken lines into paragraphs
    # Keep newlines where it looks like a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_sentences(text: str):
    """
    Robust-ish sentence splitter without extra deps.
    If you want best quality, swap this for spaCy later.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split on sentence end punctuation followed by space + capital/quote
    # Keeps punctuation attached.
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]*[A-Z])", text)

    # Fallback: if no splits happened, return whole text
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def sentence_chunk(text: str, target_chars: int = 1200, overlap_sents: int = 1):
    """
    Pack complete sentences into ~target_chars chunks.
    overlap_sents overlaps last N sentences from previous chunk.
    """
    sents = split_sentences(text)

    chunks = []
    cur = []
    cur_len = 0

    for s in sents:
        add_len = len(s) + (1 if cur else 0)
        if cur and (cur_len + add_len) > target_chars:
            chunk = " ".join(cur).strip()
            chunks.append(chunk)

            # overlap
            if overlap_sents > 0:
                cur = cur[-overlap_sents:]
                cur_len = sum(len(x) for x in cur) + max(0, len(cur) - 1)
            else:
                cur = []
                cur_len = 0

        cur.append(s)
        cur_len += add_len

    if cur:
        chunks.append(" ".join(cur).strip())

    return chunks


def tokenize_for_bm25(text: str):
    # simple word tokenizer
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing transcript files")
    ap.add_argument("--out_dir", required=True, help="Where to write indexes")
    ap.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model")
    ap.add_argument("--target_chars", type=int, default=1800)
    ap.add_argument("--overlap_chars", type=int, default=250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--overlap_sents", type=int, default=1)

    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = out_dir / "chunks.jsonl"
    bm25_path = out_dir / "bm25.pkl"
    faiss_path = out_dir / "ctt.faiss"
    meta_path = out_dir / "meta.pkl"

    print(f"[ctt] input_dir = {in_dir}")
    print(f"[ctt] out_dir   = {out_dir}")

    # 1) Read + chunk + write JSONL
    files = sorted([p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".txt", ".vtt", ".srt", ".json"]])

    if not files:
        raise SystemExit(f"No transcript files found in {in_dir}")

    all_chunks = []
    all_meta = []

    # If you have JSON transcripts, you can extend parsing here.
    for fp in tqdm(files, desc="Reading+chunking"):
        raw = fp.read_text(errors="ignore")
        text = clean_vtt_or_text(raw)
        if len(text) < 200:
            continue

        episode_id = fp.stem
        # Title guess (you can override later if your scraper saved titles)
        title = episode_id.replace("_", " ").replace("-", " ").strip()

        chunks = sentence_chunk(text, target_chars=args.target_chars, overlap_sents=args.overlap_sents)

        for i, ch in enumerate(chunks):
            # Contextual header (quick version)
            header = f"Closer To Truth transcript. Episode: {title}. Chunk {i+1}/{len(chunks)}."
            contextual = header + "\n\n" + ch

            rec = {
                "episode_id": episode_id,
                "title": title,
                "chunk_index": i,
                "text": ch,
                "contextual_text": contextual,
                "source_path": str(fp),
            }
            all_chunks.append(rec["contextual_text"])
            all_meta.append(rec)

    # Write chunks.jsonl
    with chunk_path.open("w", encoding="utf-8") as f:
        for rec in all_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ctt] wrote {len(all_meta)} chunks -> {chunk_path}")

    # 2) Build BM25
    tokenized = [tokenize_for_bm25(c) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)
    print(f"[ctt] wrote BM25 -> {bm25_path}")

    # 3) Build embeddings + FAISS
    model = SentenceTransformer(args.embed_model)
    print(f"[ctt] embedding with {args.embed_model} ...")

    embs = []
    for start in tqdm(range(0, len(all_chunks), args.batch), desc="Embedding"):
        batch = all_chunks[start:start+args.batch]
        e = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embs.append(e)

    embs = np.vstack(embs).astype("float32")
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity when vectors normalized
    index.add(embs)
    faiss.write_index(index, str(faiss_path))
    print(f"[ctt] wrote FAISS -> {faiss_path}")

    # 4) Store meta in pickle for fast load
    with meta_path.open("wb") as f:
        pickle.dump(all_meta, f)
    print(f"[ctt] wrote meta -> {meta_path}")

    print("[ctt] DONE")


if __name__ == "__main__":
    main()
