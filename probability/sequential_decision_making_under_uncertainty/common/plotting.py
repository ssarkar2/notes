"""Shared plot style. Every figure is saved as .svg next to the script that makes it."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "svg.fonttype": "none",       # keep text as text: small, diffable svg
    "svg.hashsalt": "sdmu",   # deterministic ids -> stable git diffs
    "figure.figsize": (7, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

# One colour per core level (7 levels), sequential so "higher level" reads as "darker".
LEVEL_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
ACCENT = "#d95f02"


def save(fig, here: str | Path, name: str) -> Path:
    """Save `fig` as <dir of here>/<name>.svg and close it."""
    out = Path(here).resolve().parent / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg", metadata={"Date": None})
    plt.close(fig)
    print(f"wrote {out}")
    return out
