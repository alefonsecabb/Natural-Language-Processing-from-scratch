"""Generate the NLP progression diagram shown in README.md.

Run from the repo root:
    python assets/generate_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).parent / "nlp_progression_diagram.png"

milestones = [
    ("Bag-of-Words",         "1954", "#9aa0a6", "Sparse counts"),
    ("TF-IDF",               "1972", "#9aa0a6", "Weighted counts"),
    ("Word2Vec\nGloVe",      "2013", "#4285F4", "Dense embeddings"),
    ("Seq2Seq\n+ Attention", "2015", "#34A853", "Encoder-decoder"),
    ("Transformer",          "2017", "#FBBC04", "Self-attention only"),
    ("BERT",                 "2018", "#EA4335", "Bidirectional\npre-training"),
    ("GPT-3",                "2020", "#A142F4", "Few-shot\nin-context"),
    ("ChatGPT\nLLaMA",       "2023", "#F4B400", "RLHF, open weights"),
    ("GPT-4 / Claude\nGemini", "2024+", "#FF6D01", "Multimodal,\nlong context"),
]

n = len(milestones)
fig, ax = plt.subplots(figsize=(15, 5.5))
fig.patch.set_facecolor("white")

# Uniform spacing along the x-axis (one slot per milestone)
xs = list(range(n))
ax.axhline(y=0, color="#333", linewidth=2, zorder=1)
ax.set_xlim(-0.7, n - 0.3)
ax.set_ylim(-3.2, 3.2)
ax.axis("off")

for i, (name, year, color, desc) in enumerate(milestones):
    x = xs[i]
    above = (i % 2 == 0)
    y_box  = 1.6 if above else -1.6
    y_year = 0.45 if above else -0.45

    # Dot on the timeline
    ax.scatter([x], [0], s=200, color=color, zorder=3, edgecolor="white", linewidth=2)
    # Year label opposite to box
    ax.text(x, y_year, year, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333")
    # Connector line
    ax.plot([x, x], [0, y_box * 0.95], color=color, linewidth=1.5, zorder=2)
    # Colored box
    box = mpatches.FancyBboxPatch(
        (x - 0.46, y_box - 0.85), 0.92, 1.7,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="none", alpha=0.88, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y_box + 0.35, name, ha="center", va="center",
            fontsize=8.8, fontweight="bold", color="white", zorder=4)
    ax.text(x, y_box - 0.4, desc, ha="center", va="center",
            fontsize=7.0, color="white", zorder=4)

# Bracket: notebooks 01-05
def bracket(x0, x1, y, label, color):
    ax.plot([x0, x0, x1, x1], [y, y + 0.18, y + 0.18, y], color=color, linewidth=1.5)
    ax.text((x0 + x1) / 2, y + 0.42, label, ha="center", va="center",
            fontsize=9, color=color, fontweight="bold")

bracket(-0.45, 5.45, -2.65, "Notebooks 01–05 (built from scratch)", "#444")
bracket(5.55, 8.45, -2.65, "Notebook 06 (Hugging Face + RAG)", "#FF6D01")

# Title
fig.suptitle("Seven decades of Natural Language Processing — from sparse counts to chat models",
             fontsize=14, fontweight="bold", y=0.97, color="#222")

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"OK Wrote {OUT.relative_to(Path(__file__).parent.parent)}")
