"""InputBar — multiline text input + Send button with command history."""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

from . import theme


class InputBar(tk.Frame):
    """Bottom input bar with multiline Text widget, send button, and command history."""

    def __init__(
        self,
        parent: tk.Widget,
        on_submit: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, bg=theme.BG_INPUT, height=80, **kwargs)
        self.pack_propagate(False)
        self._on_submit = on_submit
        self._history: list[str] = []
        self._history_index: int = -1

        # Top border line
        tk.Frame(self, bg=theme.BORDER_COLOR, height=2).pack(fill=tk.X, side=tk.TOP)

        # Inner container
        inner = tk.Frame(self, bg=theme.BG_INPUT)
        inner.pack(fill=tk.BOTH, expand=True)

        # Prompt indicator
        tk.Label(
            inner,
            text=" Goal/Feedback: ",
            font=theme.FONT_MONO,
            fg=theme.FG_ACCENT,
            bg=theme.BG_INPUT,
        ).pack(side=tk.LEFT, padx=(8, 0), anchor="n", pady=6)

        # Multiline text entry
        self.text = tk.Text(
            inner,
            height=3,
            font=theme.FONT_MONO,
            bg=theme.BG_DARK,
            fg=theme.FG_TEXT,
            insertbackground=theme.FG_TEXT,
            selectbackground=theme.FG_ACCENT,
            selectforeground=theme.BG_DARK,
            relief=tk.SOLID,
            borderwidth=1,
            highlightthickness=1,
            highlightcolor=theme.FG_ACCENT,
            highlightbackground=theme.BORDER_COLOR,
            wrap=tk.WORD,
            padx=4,
            pady=4,
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=theme.PAD, pady=6)

        # Ctrl+Enter to submit, plain Enter for newline
        self.text.bind("<Control-Return>", self._handle_submit)
        self.text.bind("<Control-KP_Enter>", self._handle_submit)

        # Send button
        self.send_btn = tk.Button(
            inner,
            text=" Send \n Ctrl+↵ ",
            font=theme.FONT_MONO_SMALL,
            bg=theme.BG_BUTTON_ACCENT,
            fg=theme.FG_BUTTON_ACCENT,
            activebackground=theme.FG_SUCCESS,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=self._submit,
        )
        self.send_btn.pack(side=tk.RIGHT, padx=(theme.PAD, 8), pady=6)

        # Placeholder text
        self._placeholder = "Enter goal, /command, or live feedback... (Ctrl+Enter to send)"
        self._has_placeholder = False
        self._show_placeholder()
        self.text.bind("<FocusIn>", self._clear_placeholder)
        self.text.bind("<FocusOut>", self._maybe_show_placeholder)

    def _show_placeholder(self) -> None:
        content = self.text.get("1.0", tk.END).strip()
        if not content:
            self.text.insert("1.0", self._placeholder)
            self.text.configure(fg=theme.FG_DIM)
            self._has_placeholder = True

    def _clear_placeholder(self, event=None) -> None:
        if self._has_placeholder:
            self.text.delete("1.0", tk.END)
            self.text.configure(fg=theme.FG_TEXT)
            self._has_placeholder = False

    def _maybe_show_placeholder(self, event=None) -> None:
        content = self.text.get("1.0", tk.END).strip()
        if not content:
            self._show_placeholder()

    def _handle_submit(self, event=None) -> str:
        self._submit()
        return "break"  # Prevent default Ctrl+Enter behavior

    def _submit(self) -> None:
        text = self.text.get("1.0", tk.END).strip()
        if not text or text == self._placeholder:
            return

        # Add to history
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._history_index = -1

        self.text.delete("1.0", tk.END)
        self._has_placeholder = False

        if self._on_submit:
            self._on_submit(text)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable input."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.text.configure(state=state)
        self.send_btn.configure(state=state)

    def focus_entry(self) -> None:
        """Focus the text widget."""
        self.text.focus_set()

    def get_text(self) -> str:
        """Get current text content."""
        content = self.text.get("1.0", tk.END).strip()
        if content == self._placeholder:
            return ""
        return content
