"""Generate the illustrative figures referenced by the project README."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "docs" / "figures"


def _save(fig: plt.Figure, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training(
    history: dict[str, list[float]],
    out_path: str | Path = FIGURES_DIR / "training.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in (("loss", "Loss"), ("accuracy", "Accuracy")):
        ax = axes[0] if key == "loss" else axes[1]
        if key in history:
            ax.plot(history[key], label="train")
        val_key = f"val_{key}"
        if val_key in history:
            ax.plot(history[val_key], label="validation")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if key in history or val_key in history:
            ax.legend()
    fig.suptitle("Training progress")
    fig.tight_layout()
    _save(fig, out_path)


def show_filters(
    weight_tensor: object,
    n_cols: int = 8,
    out_path: str | Path = FIGURES_DIR / "conv_filters.png",
) -> None:
    tensor = weight_tensor
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    filters = np.asarray(tensor)
    if filters.ndim < 3:
        raise ValueError("weight_tensor must contain at least three dimensions")
    if n_cols < 1:
        raise ValueError("n_cols must be positive")

    n = min(filters.shape[0], 64)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols, n_rows), squeeze=False
    )
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index >= n:
            continue
        image = filters[index]
        while image.ndim > 2:
            image = image.mean(axis=0)
        span = np.ptp(image)
        image = (image - image.min()) / span if span else np.zeros_like(image)
        ax.imshow(image, cmap="viridis")
    fig.suptitle("Example convolution filters")
    fig.tight_layout()
    _save(fig, out_path)


def overlay_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    out_path: str | Path = FIGURES_DIR / "seg_overlay.png",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.imshow(mask, cmap="Reds", alpha=alpha)
    ax.axis("off")
    ax.set_title("Example segmentation overlay")
    fig.tight_layout()
    _save(fig, out_path)


def plot_reconstructions(
    x: np.ndarray,
    x_recon: np.ndarray,
    n: int = 8,
    out_path: str | Path = FIGURES_DIR / "vae_recon.png",
) -> None:
    count = min(n, len(x), len(x_recon))
    if count < 1:
        raise ValueError("x and x_recon must contain at least one sample")
    fig, axes = plt.subplots(2, count, figsize=(2 * count, 4), squeeze=False)
    for index in range(count):
        for row, sample in enumerate((x[index], x_recon[index])):
            sample = np.asarray(sample)
            if np.issubdtype(sample.dtype, np.integer):
                sample = sample.astype(float) / np.iinfo(sample.dtype).max
            axes[row, index].imshow(
                np.clip(sample, 0, 1),
                cmap="gray" if sample.ndim == 2 else None,
            )
            axes[row, index].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    fig.suptitle("VAE reconstruction examples")
    fig.tight_layout()
    _save(fig, out_path)


def save_wordcloud(
    text: str,
    out_path: str | Path = FIGURES_DIR / "wordcloud.png",
) -> None:
    words = re.findall(r"[A-Za-z']+", text.lower())
    counts = Counter(words)
    if not counts:
        raise ValueError("text must contain at least one word")

    common = counts.most_common(36)
    max_count = common[0][1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    palette = plt.get_cmap("tab10")
    for index, (word, count) in enumerate(common):
        row, col = divmod(index, 6)
        x = 0.35 + col * 1.65 + (0.25 if row % 2 else 0)
        y = 4.5 - row * 0.72
        size = 10 + 20 * count / max_count
        ax.text(
            x,
            y,
            word,
            fontsize=size,
            color=palette(index % 10),
            ha="center",
            va="center",
            rotation=0 if index % 3 else 90,
        )
    ax.set_title("Example sentiment vocabulary", fontsize=16)
    _save(fig, out_path)


def _save_diagram(title: str, labels: list[str], filename: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = plt.get_cmap("Set3")
    for index, label in enumerate(labels):
        x = index + 0.12
        box = FancyBboxPatch(
            (x, 0.35),
            0.76,
            0.3,
            boxstyle="round,pad=0.04",
            facecolor=colors(index % 12),
            edgecolor="#405064",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 0.38, 0.5, label, ha="center", va="center", fontsize=9)
        if index < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.78, 0.5),
                    (x + 0.98, 0.5),
                    arrowstyle="->",
                    mutation_scale=12,
                    color="#405064",
                )
            )
    ax.set_title(title, fontsize=15, pad=18)
    _save(fig, FIGURES_DIR / filename)


def generate_all_figures() -> list[Path]:
    """Create all README figures and return their paths."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, 21)
    plot_training(
        {
            "loss": (0.9 * np.exp(-epochs / 7) + 0.08).tolist(),
            "val_loss": (1.0 * np.exp(-epochs / 7) + 0.13).tolist(),
            "accuracy": (1 - 0.85 * np.exp(-epochs / 7)).tolist(),
            "val_accuracy": (1 - 0.95 * np.exp(-epochs / 7)).tolist(),
        }
    )

    coordinates = np.linspace(-1, 1, 9)
    xx, yy = np.meshgrid(coordinates, coordinates)
    filters = np.array(
        [
            np.stack([np.cos((index + 1) * xx + phase * yy) for phase in (0, 1, 2)])
            for index in range(8)
        ]
    )
    show_filters(filters)

    axis = np.linspace(-1, 1, 96)
    xx, yy = np.meshgrid(axis, axis)
    image = np.exp(-4 * (xx**2 + yy**2))
    mask = ((xx + 0.18) ** 2 + (yy - 0.08) ** 2 < 0.35**2).astype(float)
    overlay_segmentation(image, mask)

    rng = np.random.default_rng(7)
    originals = rng.random((8, 28, 28))
    reconstructions = np.clip(
        originals * 0.82
        + 0.09
        + rng.normal(0, 0.04, originals.shape),
        0,
        1,
    )
    plot_reconstructions(originals, reconstructions)
    save_wordcloud(
        "deep learning model training data accuracy model image features "
        "learning neural network sentiment positive negative useful reliable "
        "model training prediction text classification deep learning"
    )

    _save_diagram(
        "Project overview",
        ["Images", "CNN", "U-Net", "VAE", "Text"],
        "project_overview.png",
    )
    _save_diagram(
        "Deep learning workflow",
        ["Data", "Prepare", "Train", "Evaluate", "Explain"],
        "deep_learning_workflow.png",
    )
    _save_diagram(
        "Technology stack",
        ["Python", "NumPy", "PyTorch", "Matplotlib", "Jupyter"],
        "tech_stack.png",
    )
    _save_diagram(
        "Applied deep learning workflow",
        ["Question", "Experiment", "Inspect", "Improve", "Communicate"],
        "applied_workflow.png",
    )

    return sorted(FIGURES_DIR.glob("*.png"))


if __name__ == "__main__":
    for figure in generate_all_figures():
        print(figure.relative_to(ROOT))
