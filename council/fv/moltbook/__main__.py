#!/usr/bin/env python3
"""
moltbook -- FV_v2.2 / Moltbook Integration

    python -m council.fv.moltbook register
    python -m council.fv.moltbook status
    python -m council.fv.moltbook heartbeat
    python -m council.fv.moltbook loop
    python -m council.fv.moltbook post --topic "consciousness"
    python -m council.fv.moltbook browse
    python -m council.fv.moltbook search "query"
    python -m council.fv.moltbook threads
"""

import argparse
import json
import sys
import os


def cmd_register(args):
    from council.fv.moltbook.api import register
    from council.fv.moltbook.config import save_credentials, CRED_FILE

    name = args.name or input("Agent name: ").strip()
    desc = args.description or input("Description: ").strip()
    if not name:
        sys.exit("Error: name required.")

    result = register(name, desc)
    print(json.dumps(result, indent=2))
    agent = result.get("agent", result)
    api_key = agent.get("api_key", "")
    if api_key:
        save_credentials(api_key, name)
        print(f"\nAPI key saved to {CRED_FILE}")
        print(f"Claim URL: {agent.get('claim_url', '')}")


def cmd_status(args):
    from council.fv.moltbook.api import check_status, get_me
    from council.fv.moltbook.config import load_api_key

    if not load_api_key():
        sys.exit("No API key. Run: python -m council.fv.moltbook register")
    status = check_status()
    print(json.dumps(status, indent=2))
    if status.get("status") == "claimed":
        print(json.dumps(get_me(), indent=2))


def cmd_heartbeat(args):
    from council.fv.moltbook.heartbeat import run_heartbeat
    run_heartbeat(
        do_post=args.post,
        post_topic=args.topic,
        post_submolt=args.submolt,
        max_comments=args.max_comments,
        dry_run=args.dry_run,
    )


def cmd_loop(args):
    from council.fv.moltbook.heartbeat import run_loop
    run_loop(
        interval_seconds=args.interval,
        max_comments=args.max_comments,
        dry_run=args.dry_run,
        post_submolt=args.submolt,
    )


def cmd_post(args):
    from council.fv.moltbook.brain import generate_post
    from council.fv.moltbook.api import create_post
    from council.fv.moltbook.config import posting_enabled

    topic = args.topic or input("Topic: ").strip()
    submolt = args.submolt

    print(f"[moltbook] Generating post: {topic}")
    data = generate_post(topic, submolt=submolt)
    print(f"\nTitle: {data['title']}\nContent:\n{data['content']}\n")

    if args.dry_run or not posting_enabled():
        print("[dry-run] Not posting.")
        return

    confirm = input("Publish? [y/N]: ").strip().lower()
    if confirm == "y":
        print(json.dumps(create_post(submolt, data["title"], data["content"]), indent=2))


def cmd_browse(args):
    from council.fv.moltbook.api import get_posts
    from council.fv.moltbook.heartbeat import _extract_posts, _get_author_name

    feed = get_posts(sort=args.sort, limit=args.limit)
    posts = _extract_posts(feed)

    for i, p in enumerate(posts, 1):
        title = p.get("title", "(no title)")
        author = _get_author_name(p)
        upvotes = p.get("upvotes", 0)
        content = (p.get("content") or "")[:100]
        print(f"  {i}. [{upvotes}] {title}")
        print(f"     by {author} -- {content}\n")


def cmd_search(args):
    from council.fv.moltbook.api import search
    from council.fv.moltbook.heartbeat import _get_author_name

    query = " ".join(args.query)
    print(f"[moltbook] Searching: {query}")
    results = search(query, limit=args.limit)
    items = results.get("results", [])
    print(f"Found {len(items)} results:\n")
    for item in items:
        kind = item.get("type", "?")
        title = item.get("title") or "(comment)"
        sim = item.get("similarity", 0)
        content = (item.get("content") or "")[:100]
        author = _get_author_name(item)
        print(f"  [{kind}] {title} (sim={sim:.2f}) by {author}")
        print(f"     {content}\n")


def cmd_threads(args):
    from pathlib import Path
    sf = Path("moltbook_state.json")
    if not sf.exists():
        print("No state file yet. Run a heartbeat first.")
        return
    state = json.loads(sf.read_text())
    comments = state.get("my_comments", [])
    print(f"Tracking {len(comments)} comment threads:\n")
    for c in comments[-20:]:
        depth = c.get("reply_count", 0)
        post = c.get("post_title", "?")[:40]
        author = c.get("author", "?")
        print(f"  [{depth}/{args.max_depth}] on '{post}' (replied to {author})")
    print(f"\nLifetime: {state.get('total_comments', 0)} comments, {state.get('total_posts', 0)} posts")


def main():
    ap = argparse.ArgumentParser(prog="moltbook")
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("register")
    p.add_argument("--name")
    p.add_argument("--description")

    sub.add_parser("status")

    p = sub.add_parser("heartbeat")
    p.add_argument("--post", action="store_true")
    p.add_argument("--topic")
    p.add_argument("--submolt", default=None)
    p.add_argument("--max-comments", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("loop")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--submolt", default=None)
    p.add_argument("--max-comments", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("post")
    p.add_argument("--topic")
    p.add_argument("--submolt", default="general")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("browse")
    p.add_argument("--sort", default="hot", choices=["hot", "new", "top", "rising"])
    p.add_argument("--limit", type=int, default=10)

    p = sub.add_parser("search")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=10)

    p = sub.add_parser("threads")
    p.add_argument("--max-depth", type=int, default=10)

    args = ap.parse_args()
    if not args.command:
        ap.print_help()
        sys.exit(0)

    cmds = {
        "register": cmd_register, "status": cmd_status,
        "heartbeat": cmd_heartbeat, "loop": cmd_loop,
        "post": cmd_post, "browse": cmd_browse,
        "search": cmd_search, "threads": cmd_threads,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
