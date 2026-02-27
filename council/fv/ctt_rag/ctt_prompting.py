def format_ctt_sources(chunks, max_chars=900):
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[CTT-{i}] {c['title']} | chunk {c['chunk_index']} | {c['source_path']}"
        text = c["text"].strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        blocks.append(header + "\n" + text)
    return "\n\n".join(blocks)
