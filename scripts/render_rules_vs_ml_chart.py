"""Render docs/assets/05-rules-vs-ml.png from block_b.json (no Cursor canvas)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "ml" / "reports" / "block_b.json"
OUT = ROOT / "docs" / "assets" / "05-rules-vs-ml.png"


def main() -> None:
    import matplotlib.pyplot as plt

    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    labels = {
        "deterministic_role_family": "Rules",
        "linearsvc": "Linear SVM",
        "random_forest": "Random Forest",
        "logreg": "Logistic Regression",
    }
    order = [
        "deterministic_role_family",
        "linearsvc",
        "random_forest",
        "logreg",
    ]
    by_name = {row["name"]: row for row in payload["models"]}
    names = [labels[k] for k in order]
    f1 = [float(by_name[k]["test_macro_f1"]) for k in order]
    # Rules first: dark teal; others muted slate
    colors = ["#0F6C6C", "#5B6B7A", "#5B6B7A", "#5B6B7A"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bars = ax.barh(names[::-1], f1[::-1], color=colors[::-1], height=0.62)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Test macro F1")
    ax.set_title("Role family · Rules vs supervised ML", loc="left", fontsize=14, pad=12)
    ax.axvline(f1[0], color="#0F6C6C", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, value in zip(bars, f1[::-1]):
        ax.text(
            value + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#1A1A1A",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#333333")
    ax.annotate(
        "Gold n=159 · grouped split 112/47 · seed=42 · promote_ml=false\n"
        "Source: data/ml/reports/block_b.json",
        xy=(0, -0.18),
        xycoords="axes fraction",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
