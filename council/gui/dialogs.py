"""Dialogs — Safety approval confirmation dialog."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext
from typing import Optional

from . import theme


def _block_text_edits(event: tk.Event) -> Optional[str]:
    navigation_keys = {
        "Left", "Right", "Up", "Down",
        "Home", "End", "Prior", "Next",
        "Shift_L", "Shift_R", "Control_L", "Control_R",
    }
    ctrl_pressed = (event.state & 0x4) != 0
    if ctrl_pressed and event.keysym.lower() in {"c", "a"}:
        return None
    if event.keysym in navigation_keys:
        return None
    return "break"


def _make_copyable_readonly(text_widget: tk.Text) -> None:
    text_widget.configure(state=tk.NORMAL, insertwidth=0, takefocus=True)
    text_widget.bind("<Key>", _block_text_edits, add="+")
    text_widget.bind("<<Paste>>", lambda _: "break", add="+")
    text_widget.bind("<<Cut>>", lambda _: "break", add="+")


class SafetyApprovalDialog(tk.Toplevel):
    """Modal dialog for reviewing and approving/denying a safety alert."""

    def __init__(
        self,
        parent: tk.Widget,
        agent: str,
        matched_text: str,
        context: str,
        on_approve: Optional[callable] = None,
        on_deny: Optional[callable] = None,
    ):
        super().__init__(parent)
        self.title("Safety Review Required")
        self.geometry("600x400")
        self.configure(bg=theme.BG_DARK)
        self.transient(parent)
        self.grab_set()

        self._on_approve = on_approve
        self._on_deny = on_deny

        # Warning header
        header = tk.Frame(self, bg=theme.FG_WARNING)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=f"  \u26a0  Agent '{agent}' wants to execute a potentially destructive action",
            font=theme.FONT_TITLE,
            fg=theme.BG_DARK,
            bg=theme.FG_WARNING,
            anchor="w",
        ).pack(fill=tk.X, padx=theme.PAD, pady=theme.PAD)

        # Matched text
        tk.Label(
            self,
            text=f"Detected: {matched_text}",
            font=theme.FONT_MONO,
            fg=theme.FG_ERROR,
            bg=theme.BG_DARK,
            anchor="w",
        ).pack(fill=tk.X, padx=theme.PAD * 2, pady=(theme.PAD, 0))

        # Context
        tk.Label(
            self,
            text="Context:",
            font=theme.FONT_MONO,
            fg=theme.FG_DIM,
            bg=theme.BG_DARK,
            anchor="w",
        ).pack(fill=tk.X, padx=theme.PAD * 2, pady=(theme.PAD, 0))

        ctx_text = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            font=theme.FONT_MONO_SMALL,
            bg=theme.BG_PANEL,
            fg=theme.FG_TEXT,
            height=10,
            borderwidth=0,
            highlightthickness=0,
        )
        ctx_text.pack(fill=tk.BOTH, expand=True, padx=theme.PAD * 2, pady=theme.PAD)
        ctx_text.insert("1.0", context)
        _make_copyable_readonly(ctx_text)

        # Buttons
        btn_frame = tk.Frame(self, bg=theme.BG_DARK)
        btn_frame.pack(fill=tk.X, padx=theme.PAD, pady=theme.PAD)

        tk.Button(
            btn_frame,
            text="  Deny  ",
            font=theme.FONT_MONO,
            bg=theme.FG_ERROR,
            fg=theme.BG_DARK,
            relief=tk.FLAT,
            padx=14,
            pady=4,
            command=self._handle_deny,
        ).pack(side=tk.RIGHT, padx=theme.PAD)

        tk.Button(
            btn_frame,
            text="  Approve  ",
            font=theme.FONT_MONO,
            bg=theme.FG_SUCCESS,
            fg=theme.BG_DARK,
            relief=tk.FLAT,
            padx=14,
            pady=4,
            command=self._handle_approve,
        ).pack(side=tk.RIGHT, padx=theme.PAD)

    def _handle_approve(self) -> None:
        if self._on_approve:
            self._on_approve()
        self.destroy()

    def _handle_deny(self) -> None:
        if self._on_deny:
            self._on_deny()
        self.destroy()
