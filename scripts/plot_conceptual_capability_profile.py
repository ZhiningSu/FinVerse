from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AXES = [
    "Market State\nUnderstanding",
    "Short-Horizon\nForecasting",
    "Long-Horizon\nImagination",
    "Cross-Asset\nReasoning",
    "Regime\nAwareness",
    "Counterfactual\nSimulation",
    "Strategy\nEvaluation",
    "Risk\nAwareness",
    "Sample\nEfficiency",
    "Agent\nReadiness",
]

PROFILES = {
    "FinVerse Cognitive Core": [0.96, 0.94, 0.92, 0.91, 0.90, 0.95, 0.94, 0.89, 0.93, 0.96],
    "Direct Forecasting Model": [0.62, 0.84, 0.48, 0.42, 0.35, 0.18, 0.30, 0.28, 0.46, 0.25],
    "Vanilla World Model": [0.66, 0.70, 0.76, 0.40, 0.44, 0.68, 0.55, 0.52, 0.69, 0.58],
}

COLORS = {
    "FinVerse Cognitive Core": "#0B4EA2",
    "Direct Forecasting Model": "#F59E0B",
    "Vanilla World Model": "#16A34A",
}


def plot(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10.5, 9.2), subplot_kw={"polar": True})
    for name, values in PROFILES.items():
        closed = values + values[:1]
        is_ours = name == "FinVerse Cognitive Core"
        ax.plot(
            angles,
            closed,
            label=name,
            color=COLORS[name],
            linewidth=4.0 if is_ours else 2.0,
            alpha=0.95 if is_ours else 0.58,
            zorder=5 if is_ours else 2,
        )
        ax.fill(
            angles,
            closed,
            color=COLORS[name],
            alpha=0.16 if is_ours else 0.04,
            zorder=1,
        )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(color="#B7BEC8", alpha=0.72)
    ax.set_title("Conceptual Capability Profile", fontsize=17, pad=30, weight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.36, -0.04), frameon=True, fontsize=10)
    fig.text(
        0.5,
        0.025,
        "Conceptual illustration only: scores indicate intended agent-world-model capabilities, not empirical benchmark results.",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "conceptual_capability_profile.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "conceptual_capability_profile.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot(Path("paper/figures"))
