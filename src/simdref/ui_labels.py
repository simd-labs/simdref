"""Single source of truth for UI labels + keymap shared by TUI and web SPA.

``UI_LABELS`` and ``KEYMAP`` are the canonical vocabulary; ``as_json_dict``
serialises them for consumers outside this module. The TUI (``tui.py``)
imports directly; the web SPA, which lives in the ``simdref-web`` repo,
gets its own copy and is responsible for keeping it in sync.

Add new labels or bindings here first, then reference them from ``tui.py``.
A parity test enforces coverage on the TUI side — see
``tests/test_ui_labels_parity.py``.
"""

from __future__ import annotations

from typing import Final


# Canonical labels. Plan rule: ends the "Agg"/"asm"/"modeled" drift.
UI_LABELS: Final[dict[str, str]] = {
    # Kinds
    "kind_intrinsic": "Intrinsics",
    "kind_instruction": "Instructions",
    # Microarch aggregation
    "aggregation": "Aggregation",
    # Perf source labels. Modeled data is ARM/RISC-V (llvm-mca); every x86
    # row is measured.
    "source_measured": "Measured",
    "source_modeled": "Modeled (ARM/RISC-V)",
    # Toolbar labels
    "arch": "Arch",
    "isa": "ISA",
    "category": "Category",
    "perf": "Perf",
    # Status text
    "no_matches": "No matches",
    "results_for": "results for",
    # Toolbar checkboxes
    "toggle_perf": "perf",
    "toggle_docs": "docs",
    "toggle_modeled": "include modeled",
}


# Keymap: action -> (primary_key, description). Both surfaces must cover
# every key in this table (parity test enforces).
KEYMAP: Final[dict[str, tuple[str, str]]] = {
    "focus_search": ("/", "Focus search"),
    "focus_search_alt": ("ctrl+k", "Focus search"),
    "toggle_filter_drawer": ("f", "Toggle filter drawer"),
    "cycle_kind": ("k", "Cycle kind"),
    "cycle_arch": ("a", "Cycle arch"),
    "next_result": ("j", "Next result"),
    "prev_result": ("k", "Previous result"),
    "open_detail": ("enter", "Open detail section"),
    "switch_tab": ("tab", "Switch Search/Annotate"),
    "help": ("?", "Help"),
    "quit_or_close": ("escape", "Quit / close drawer"),
}


def keymap_actions() -> frozenset[str]:
    """Return the canonical action-name set used for parity checks."""
    return frozenset(KEYMAP.keys())


def as_json_dict() -> dict:
    """Projection used by the web template injector."""
    return {
        "labels": dict(UI_LABELS),
        "keymap": {action: list(pair) for action, pair in KEYMAP.items()},
    }
