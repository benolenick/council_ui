"""Dark theme colors, fonts, and style constants for the Council GUI."""

from __future__ import annotations

# --- Color Palette ---
BG_DARK = "#1e1e2e"        # Main background
BG_PANEL = "#282840"        # Panel background
BG_INPUT = "#313152"        # Input field background
BG_STATUS = "#11111b"       # Status bar background
BG_BUTTON = "#45475a"       # Button background
BG_BUTTON_HOVER = "#585b70" # Button hover
BG_BUTTON_ACCENT = "#89b4fa" # Accent button (Approve)
BG_SCROLLBAR = "#45475a"

FG_TEXT = "#cdd6f4"         # Primary text
FG_DIM = "#6c7086"          # Dimmed text
FG_ACCENT = "#89b4fa"       # Accent/link color
FG_SUCCESS = "#a6e3a1"      # Success green
FG_WARNING = "#f9e2af"      # Warning yellow
FG_ERROR = "#f38ba8"        # Error red
FG_BUTTON_ACCENT = "#1e1e2e" # Text on accent button

BORDER_COLOR = "#45475a"

# Per-agent highlight colors
AGENT_COLORS = {
    "codex":  "#f9e2af",    # Yellow — Builder
    "claude": "#89b4fa",    # Blue — Architect
    "gemini": "#f38ba8",    # Red/Pink — Skeptic
    "qwen":   "#a6e3a1",    # Green — Synthesizer
}

# --- Fonts ---
FONT_FAMILY = "monospace"
FONT_SIZE = 10
FONT_SIZE_SMALL = 9
FONT_SIZE_TITLE = 12
FONT_SIZE_STATUS = 9

FONT_MONO = (FONT_FAMILY, FONT_SIZE)
FONT_MONO_SMALL = (FONT_FAMILY, FONT_SIZE_SMALL)
FONT_TITLE = (FONT_FAMILY, FONT_SIZE_TITLE, "bold")
FONT_STATUS = (FONT_FAMILY, FONT_SIZE_STATUS)

# --- Layout ---
PAD = 4
PAD_SMALL = 2
PANEL_MIN_WIDTH = 300
PANEL_MIN_HEIGHT = 200
STATUS_HEIGHT = 28
INPUT_HEIGHT = 36
