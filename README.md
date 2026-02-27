# Council

CLI tool that orchestrates three AI coding agents (Claude Code, Codex, Gemini CLI) in a structured, human-supervised council workflow.

## How It Works

Council runs a multi-phase protocol:

1. **Builder** (Codex) generates a code change proposal
2. **Architect** (Claude) reviews the proposal
3. **Skeptic** (Gemini) reviews the proposal
4. **Builder** revises based on feedback (if needed)
5. **Synthesizer** produces a decision packet with approval hash

No code is applied without explicit human approval.

## Requirements

- Python 3.12+
- [Claude Code](https://claude.ai/claude-code) — `claude` CLI
- [Codex](https://github.com/openai/codex) — `codex` CLI
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) — `gemini` CLI

No external Python dependencies required (pure stdlib).

## Quick Start

```bash
# Initialize a workspace
python3 council.py init --workspace /path/to/your/project

# Run a council session
python3 council.py run --goal "Add input validation to the login form" \
                       --workspace /path/to/your/project

# Or use interactive mode
python3 council.py chat --workspace /path/to/your/project
```

## Commands

| Command   | Description                            |
|-----------|----------------------------------------|
| `init`    | Initialize `.council/` in a workspace  |
| `doctor`  | Verify 4-agent runtime prerequisites    |
| `run`     | One-shot council execution             |
| `chat`    | Interactive REPL                       |
| `approve` | Approve and apply a decision           |
| `status`  | Show recent council runs               |
| `logs`    | View run logs and transcripts          |

## Interactive Mode (chat)

```
council> /goal Refactor the auth module to use JWT
council> /constraints Must be backwards compatible with session tokens
council> /run
council> /show
council> /approve a1b2c3d4
council> /exit
```

## Configuration

Config lives in `.council/config.json`. Key settings:

```json
{
  "agents": {
    "builder":   { "name": "codex",  "command": "codex",  "timeout": 120 },
    "architect": { "name": "claude", "command": "claude", "timeout": 120 },
    "skeptic":   { "name": "gemini", "command": "gemini", "timeout": 120 }
  },
  "max_review_rounds": 2,
  "test_commands": ["pytest", "npm test"]
}
```

## Safety

- All file operations scoped to workspace root
- Secret file detection (`.env`, `*.key`, etc.)
- Risk tier classification (LOW/MEDIUM/HIGH)
- Patch approval requires typing `approve <hash>`
- Denied path lists prevent touching sensitive files
- Test commands checked against allowlist

## Run Artifacts

Each run creates `.council/runs/<timestamp>/` containing:
- `council.log` — structured log
- `transcript.jsonl` — phase-by-phase transcript
- `proposal.json` — builder's proposal
- `review_*.json` — reviewer feedback
- `decision.json` — final decision packet
