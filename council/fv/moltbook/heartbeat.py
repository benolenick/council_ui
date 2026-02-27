"""
moltbook.heartbeat
──────────────────
Aggressive participation loop that fully exercises FV's pipeline.

Features:
  - Comments every ~25s (within Moltbook's 20s cooldown)
  - Tracks threads: checks if anyone replied to our comments, replies back (up to 10 deep)
  - Saves all interactions to FV's STM so it remembers who it talked to
  - Posts original science content on schedule
  - Respects all .env settings
  - Detects suspension / rate-limit and stops immediately
"""

from __future__ import annotations

import json
import time
import random
import traceback
from pathlib import Path
from typing import Optional, List

from council.fv.moltbook import api
from council.fv.moltbook.api import SuspendedError, RateLimitError, VerificationFailedError
from council.fv.moltbook.brain import (
    generate_comment,
    generate_post,
    generate_thread_reply,
    decide_engagement,
)
from council.fv.moltbook.config import (
    posting_enabled,
    moltbook_mode,
    default_submolt,
    max_actions_per_run,
    min_action_gap_seconds,
    ignore_seen,
    max_thread_depth,
    comment_interval_seconds,
)


# ── Persistent state file ──

STATE_FILE = Path(__file__).resolve().parent.parent / "moltbook_state.json"

SCIENCE_TOPICS = [
    "the hard problem of consciousness and why it resists reductionist explanation",
    "whether the fine-tuning of physical constants implies design or selection bias",
    "how the Many-Worlds interpretation changes our understanding of probability",
    "whether free will is compatible with what neuroscience tells us about decisions",
    "emergence vs reductionism — can mind be explained by physics alone?",
    "the cosmological argument and whether the universe needs a cause",
    "what quantum mechanics reveals about the nature of reality",
    "the relationship between mathematics and physical reality",
    "whether artificial intelligence could ever be truly conscious",
    "panpsychism as a serious theory of consciousness",
    "the measurement problem in quantum mechanics",
    "whether the laws of physics are discovered or invented",
    "the problem of other minds and how we know anything is conscious",
    "entropy and the arrow of time",
    "the role of observation in shaping reality",
]


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "last_heartbeat": None,
        "engaged_post_ids": [],        # posts we've commented on
        "my_comments": [],             # {post_id, comment_id, author, timestamp, reply_count}
        "last_post_time": None,
        "actions_this_run": 0,
        "total_posts": 0,
        "total_comments": 0,
    }


def _save_state(state: dict):
    import os as _os
    tmp = str(STATE_FILE) + ".tmp"
    Path(tmp).write_text(json.dumps(state, indent=2, default=str))
    _os.replace(tmp, str(STATE_FILE))


def _extract_posts(feed_response) -> list:
    """Normalize different API response shapes into a list of post dicts."""
    if isinstance(feed_response, list):
        return feed_response
    if isinstance(feed_response, dict):
        for key in ("posts", "data", "results"):
            if isinstance(feed_response.get(key), list):
                return feed_response[key]
    return []


def _extract_comments(response) -> list:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        for key in ("comments", "data", "results"):
            if isinstance(response.get(key), list):
                return response[key]
    return []


def _get_author_name(obj: dict) -> str:
    author = obj.get("author", {})
    if isinstance(author, dict):
        return author.get("name", "?")
    return str(author) if author else "?"


# ──────────────────────────────────────────────────────────
# Phase 1: Check threads — find replies to our old comments
# ──────────────────────────────────────────────────────────

def _check_threads(state: dict, dry_run: bool, verbose: bool) -> int:
    """
    Look at posts where we previously commented. If someone replied
    to our comment, reply back (up to max_thread_depth).

    Returns number of actions taken.
    Raises SuspendedError/RateLimitError to abort immediately.
    """
    max_depth = max_thread_depth()
    gap = comment_interval_seconds()
    actions = 0
    my_comments = state.get("my_comments", [])

    if not my_comments:
        return 0

    if verbose:
        print(f"[threads] Checking {len(my_comments)} tracked comments for replies...")

    updated_comments = []

    for entry in my_comments:
        post_id = entry.get("post_id")
        my_comment_id = entry.get("comment_id")
        reply_count = entry.get("reply_count", 0)

        if reply_count >= max_depth:
            if verbose:
                print(f"  [skip] Thread on post {post_id[:8]}... hit max depth ({max_depth})")
            updated_comments.append(entry)
            continue

        try:
            comments_resp = api.get_comments(post_id, sort="new")
            all_comments = _extract_comments(comments_resp)
        except Exception as e:
            if verbose:
                print(f"  [err] Could not fetch comments for {post_id[:8]}...: {e}")
            updated_comments.append(entry)
            continue

        # Find replies to our comment (comments with parent_id == my_comment_id)
        new_replies = []
        for c in all_comments:
            c_id = c.get("id", "")
            if c_id == my_comment_id:
                continue
            parent = c.get("parent_id") or c.get("parentId")
            if parent == my_comment_id:
                new_replies.append(c)

        if not new_replies:
            updated_comments.append(entry)
            continue

        # We have replies! Generate a response to the most recent one
        latest = new_replies[-1]
        their_author = _get_author_name(latest)
        their_text = latest.get("content", "")

        if verbose:
            print(f"  [reply] {their_author} replied to us on post {post_id[:8]}...")
            print(f"          They said: {their_text[:80]}...")

        # Build thread history for context
        thread_history = []
        for c in all_comments:
            thread_history.append({
                "author": _get_author_name(c),
                "content": c.get("content", ""),
            })

        # Get post title for context
        post_title = entry.get("post_title", "")
        if not post_title:
            try:
                post_data = api.get_post(post_id)
                if isinstance(post_data, dict):
                    post_title = post_data.get("title") or post_data.get("data", {}).get("title", "")
            except Exception:
                pass

        # Generate reply through FV pipeline
        try:
            reply_text = generate_thread_reply(
                post_id=post_id,
                post_title=post_title,
                their_comment=their_text,
                their_author=their_author,
                thread_history=thread_history,
            )

            if verbose:
                print(f"          Our reply: {reply_text[:80]}...")

            if not dry_run:
                # This will raise SuspendedError/RateLimitError if needed
                result = api.create_comment(post_id, reply_text, parent_id=latest.get("id"))
                new_comment_id = None
                if isinstance(result, dict):
                    new_comment_id = result.get("id") or result.get("comment", {}).get("id")

                entry["reply_count"] = reply_count + 1
                entry["last_reply_to"] = their_author
                entry["last_reply_time"] = time.time()
                if new_comment_id:
                    entry["comment_id"] = new_comment_id

                actions += 1
                state["total_comments"] = state.get("total_comments", 0) + 1

                if verbose:
                    print(f"          Posted (thread depth: {reply_count + 1}/{max_depth})")

                time.sleep(gap)

        except (SuspendedError, RateLimitError, VerificationFailedError):
            updated_comments.append(entry)
            state["my_comments"] = updated_comments
            raise  # propagate up to abort heartbeat
        except Exception as e:
            if verbose:
                print(f"          Reply failed: {e}")

        updated_comments.append(entry)

    state["my_comments"] = updated_comments
    return actions


# ──────────────────────────────────────────────────────────
# Phase 2: Browse feed and engage with new posts
# ──────────────────────────────────────────────────────────

def _browse_and_engage(state: dict, max_comments: int, dry_run: bool, verbose: bool) -> int:
    """
    Browse feed, comment on interesting posts, upvote others.
    Returns actions taken.
    Raises SuspendedError/RateLimitError to abort immediately.
    """
    gap = comment_interval_seconds()
    actions = 0

    try:
        feed = api.get_posts(sort="new", limit=25)
        posts = _extract_posts(feed)
    except Exception as e:
        if verbose:
            print(f"[browse] Feed fetch failed: {e}")
        return 0

    if verbose:
        print(f"[browse] {len(posts)} posts in feed")

    already_engaged = set(state.get("engaged_post_ids", []))

    for post in posts:
        if actions >= max_comments:
            break

        post_id = post.get("id", "")

        if not ignore_seen() and post_id in already_engaged:
            continue

        action = decide_engagement(post)
        title = (post.get("title") or "")[:60]
        author = _get_author_name(post)

        if action == "skip":
            continue

        if action == "upvote":
            if verbose:
                print(f"  Upvoting: {title} (by {author})")
            if not dry_run:
                try:
                    api.upvote_post(post_id)
                except Exception as e:
                    if verbose:
                        print(f"     Upvote failed: {e}")
            already_engaged.add(post_id)
            continue

        if action == "comment":
            if verbose:
                print(f"  Commenting on: {title} (by {author})")

            # Fetch existing comments for context
            existing = []
            try:
                comments_resp = api.get_comments(post_id, sort="new")
                existing = _extract_comments(comments_resp)
            except Exception:
                pass

            try:
                comment_text = generate_comment(
                    post_title=post.get("title", ""),
                    post_content=post.get("content", ""),
                    author=author,
                    existing_comments=existing,
                )

                if verbose:
                    print(f"     Generated: {comment_text[:100]}...")

                if not dry_run:
                    # This will raise SuspendedError/RateLimitError
                    result = api.create_comment(post_id, comment_text)

                    # Track this comment for thread follow-up
                    comment_id = None
                    if isinstance(result, dict):
                        comment_id = result.get("id") or result.get("comment", {}).get("id")

                    state.setdefault("my_comments", []).append({
                        "post_id": post_id,
                        "post_title": post.get("title", ""),
                        "comment_id": comment_id,
                        "author": author,
                        "timestamp": time.time(),
                        "reply_count": 0,
                    })

                    # Also upvote the post
                    try:
                        api.upvote_post(post_id)
                    except Exception:
                        pass

                    actions += 1
                    state["total_comments"] = state.get("total_comments", 0) + 1

                    if verbose:
                        print(f"     Comment posted")

                    time.sleep(gap)

            except (SuspendedError, RateLimitError, VerificationFailedError):
                already_engaged.add(post_id)
                state["engaged_post_ids"] = list(already_engaged)[-500:]
                raise  # propagate up to abort heartbeat
            except Exception as e:
                if verbose:
                    print(f"     Comment failed: {e}")
                    traceback.print_exc()

            already_engaged.add(post_id)

    # Keep last 500 engaged IDs
    state["engaged_post_ids"] = list(already_engaged)[-500:]
    return actions


# ──────────────────────────────────────────────────────────
# Phase 3: Original post
# ──────────────────────────────────────────────────────────

def _make_post(state: dict, topic: Optional[str], submolt: str, dry_run: bool, verbose: bool) -> int:
    topic = topic or random.choice(SCIENCE_TOPICS)

    if verbose:
        print(f"  Generating post: {topic}")

    try:
        post_data = generate_post(topic, submolt=submolt)

        if verbose:
            print(f"     Title: {post_data['title']}")
            print(f"     Content: {post_data['content'][:120]}...")

        if not dry_run:
            result = api.create_post(submolt, post_data["title"], post_data["content"])
            state["total_posts"] = state.get("total_posts", 0) + 1
            state["last_post_time"] = time.time()

            if verbose:
                print(f"     Published")

        return 1

    except (SuspendedError, RateLimitError, VerificationFailedError):
        raise  # propagate
    except Exception as e:
        if verbose:
            print(f"     Post failed: {e}")
            traceback.print_exc()
        return 0


# ──────────────────────────────────────────────────────────
# Main heartbeat
# ──────────────────────────────────────────────────────────

def run_heartbeat(
    do_post: bool = False,
    post_topic: Optional[str] = None,
    post_submolt: Optional[str] = None,
    max_comments: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = True,
):
    """
    Run one heartbeat cycle.

    Order:
      1. Check claim status
      2. Check DMs
      3. Check threads (replies to our old comments -> reply back)
      4. Browse feed and comment on new posts
      5. Optionally post original content
    """
    state = _load_state()
    submolt = post_submolt or default_submolt()
    max_act = max_comments if max_comments is not None else max_actions_per_run()
    mode = moltbook_mode()
    enabled = posting_enabled()

    # Override dry_run if posting not enabled
    if not enabled:
        dry_run = True
        if verbose:
            print("[moltbook] POSTING_ENABLED=0 -> dry-run mode")

    def log(msg):
        if verbose:
            print(f"[moltbook] {msg}")

    log(f"Heartbeat starting (mode={mode}, max_actions={max_act}, gap={comment_interval_seconds()}s)")

    # ── 1. Status check ──
    try:
        status = api.check_status()
        s = status.get("status", "unknown")
        log(f"Status: {s}")
        if s == "pending_claim":
            log("Not yet claimed!")
            return
        # Check for suspension in status response
        if s == "suspended" or "suspend" in str(status).lower():
            log(f"Agent is suspended! Details: {json.dumps(status, default=str)[:200]}")
            log("Aborting heartbeat -- wait for suspension to expire.")
            return
    except Exception as e:
        log(f"Status check failed: {e}")
        return

    # ── 2. DMs ──
    try:
        dm = api.check_dms()
        log(f"DMs: {json.dumps(dm, default=str)[:150]}")
    except Exception as e:
        log(f"DM check: {e}")

    total_actions = 0

    try:
        # ── 3. Thread follow-ups (reply to people who replied to us) ──
        log("-- Checking threads --")
        thread_actions = _check_threads(state, dry_run, verbose)
        total_actions += thread_actions
        log(f"Thread replies: {thread_actions}")

        # ── 4. Browse + engage ──
        remaining = max_act - total_actions
        if remaining > 0 and mode in ("comment", "both"):
            log("-- Browsing feed --")
            browse_actions = _browse_and_engage(state, remaining, dry_run, verbose)
            total_actions += browse_actions
            log(f"New comments: {browse_actions}")

        # ── 5. Original post ──
        if do_post or mode in ("post", "both"):
            last_post = state.get("last_post_time")
            since_last = (time.time() - last_post) if last_post else 99999

            if since_last < 1800 and not dry_run:
                log(f"Skipping post (last post {since_last:.0f}s ago, need 1800s)")
            else:
                remaining = max_act - total_actions
                if remaining > 0:
                    log("-- Creating post --")
                    post_actions = _make_post(state, post_topic, submolt, dry_run, verbose)
                    total_actions += post_actions

    except SuspendedError as e:
        log(f"SUSPENDED -- stopping immediately: {e}")
        log("Do NOT restart until the suspension expires.")
    except RateLimitError as e:
        log(f"RATE LIMITED -- stopping. Retry after {e.retry_after}s")
    except VerificationFailedError as e:
        log(f"VERIFICATION FAILING -- stopping to avoid suspension: {e}")
        log("Check your LLM server and challenge parser before restarting.")

    # ── Save state ──
    state["last_heartbeat"] = time.time()
    state["actions_this_run"] = total_actions

    # Prune old tracked comments (keep last 200)
    if len(state.get("my_comments", [])) > 200:
        state["my_comments"] = state["my_comments"][-200:]

    _save_state(state)
    log(f"Heartbeat done. Actions: {total_actions} | Lifetime: {state.get('total_comments', 0)} comments, {state.get('total_posts', 0)} posts")


def run_loop(interval_seconds: int = 60, **kwargs):
    """
    Run heartbeat in a continuous loop.
    Default: every 60 seconds (one comment per minute pace).
    Posts original content every ~30 min.
    """
    cycle = 0
    interval = interval_seconds
    print(f"[moltbook] Starting loop (every {interval}s)")
    print(f"[moltbook] Ctrl+C to stop.\n")

    while True:
        try:
            cycle += 1
            # Post original content roughly every 30 min
            should_post = (cycle % max(1, 1800 // interval) == 1)
            run_heartbeat(do_post=should_post, **kwargs)
        except KeyboardInterrupt:
            print("\n[moltbook] Stopped.")
            break
        except Exception as e:
            print(f"[moltbook] Error: {e}")
            traceback.print_exc()

        print(f"[moltbook] Next in {interval}s...\n")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[moltbook] Stopped.")
            break
