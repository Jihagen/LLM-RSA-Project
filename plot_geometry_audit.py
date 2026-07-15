"""Generate three focused H2 geometry-audit figures.

1. PCA geometry: the two comparison layers, and nothing else.
2. Decision scores: sentence-level correct/wrong-side evidence.
3. Layer curves: separate axes for adequacy and GDV (no dual y-axis).

PCA is illustrative.  GDV and margins are always computed in the full hidden
space, never from the two-dimensional projection.
"""

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from experiments.adequacy import leave_one_out_adequacy_margins
from experiments.gdv_experiments import compute_gdv
from visual_style import (
    GRID,
    INK,
    METHOD,
    MUTED,
    OUTCOME,
    SENSE,
    apply_report_style,
)


def _load_layer(results_dir: Path, safe_model: str, word: str, layer: int):
    path = results_dir / "activations" / word / safe_model / f"layer_{layer}.h5"
    with h5py.File(path, "r") as handle:
        return handle["X"][:], handle["labels"][:]


def _save_svg_and_png(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_path.with_suffix(".svg")
    png_path = output_path.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _pca_panel(ax, X, labels, layer, gdv, fraction):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    for sense in (0, 1):
        points = coords[labels == sense]
        centroid = points.mean(axis=0)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=30,
            alpha=0.68,
            color=SENSE[sense],
            edgecolor="white",
            linewidth=0.35,
        )
        ax.scatter(
            centroid[0],
            centroid[1],
            s=150,
            marker="X",
            color=SENSE[sense],
            edgecolor=INK,
            linewidth=0.9,
            zorder=4,
        )

    c0 = coords[labels == 0].mean(axis=0)
    c1 = coords[labels == 1].mean(axis=0)
    direction = c1 - c0
    midpoint = 0.5 * (c0 + c1)
    limits = np.vstack([coords.min(axis=0), coords.max(axis=0)])
    padding = 0.10 * np.maximum(limits[1] - limits[0], 1e-9)
    xlim = (limits[0, 0] - padding[0], limits[1, 0] + padding[0])
    ylim = (limits[0, 1] - padding[1], limits[1, 1] + padding[1])
    if abs(direction[1]) > 1e-12:
        xs = np.array(xlim)
        ys = midpoint[1] - direction[0] * (xs - midpoint[0]) / direction[1]
        ax.plot(xs, ys, color=MUTED, linestyle="--", linewidth=1.3)
    elif abs(direction[0]) > 1e-12:
        ax.axvline(midpoint[0], color=MUTED, linestyle="--", linewidth=1.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"PC1 · {100 * pca.explained_variance_ratio_[0]:.1f}% variance")
    ax.set_ylabel(f"PC2 · {100 * pca.explained_variance_ratio_[1]:.1f}% variance")
    ax.set_title(f"Layer {layer}", loc="left", fontweight="bold")
    ax.text(
        0.01,
        0.01,
        f"GDV {gdv:.3f}  ·  adequate {fraction:.0%}",
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2},
    )
    ax.grid(color=GRID, linewidth=0.6)


def _make_pca_figure(layer_data, layers, model_label, word, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    for ax, layer in zip(axes, layers):
        data = layer_data[layer]
        _pca_panel(
            ax,
            data["X"],
            data["labels"],
            layer,
            data["gdv"],
            data["fraction"],
        )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=SENSE[s], label=f"sense {s}")
        for s in (0, 1)
    ] + [
        Line2D([0], [0], marker="X", linestyle="", color=MUTED, label="displayed centroid"),
        Line2D([0], [0], linestyle="--", color=MUTED, label="PCA-space boundary"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    fig.suptitle(
        f"Projected sense geometry · {model_label} · {word}",
        fontsize=14,
        fontweight="bold",
    )
    _save_svg_and_png(fig, output_path)


def _make_score_figure(layer_data, layers, model_label, word, output_path):
    fig, ax = plt.subplots(figsize=(10.8, 4.4), constrained_layout=True)
    ax.axvspan(-1, 0, color=OUTCOME["wrong"], alpha=0.10, zorder=0)
    ax.axvspan(0, 1, color=OUTCOME["correct"], alpha=0.10, zorder=0)
    ax.axvline(0, color=INK, linewidth=1.25)
    markers = {0: "o", 1: "D"}
    for row, layer in enumerate(layers):
        data = layer_data[layer]
        for sense in (0, 1):
            values = data["normalized"][data["labels"] == sense]
            jitter = np.linspace(-0.13, 0.13, len(values))
            ax.scatter(
                values,
                row + jitter,
                s=38,
                marker=markers[sense],
                color=SENSE[sense],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.4,
                label=f"sense {sense}" if row == 0 else None,
                zorder=3,
            )
        ax.text(
            1.01,
            row,
            f"{data['fraction']:.0%} correct",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-0.45, len(layers) - 0.55)
    ax.set_yticks(range(len(layers)), [f"Layer {layer}" for layer in layers])
    ax.set_xlabel(r"Leave-one-out normalized margin $\hat{M}_l$")
    ax.set_title(
        f"Which sentences fall on the correct side? · {model_label} · {word}",
        loc="left",
        fontweight="bold",
    )
    ax.text(-0.98, len(layers) - 0.63, "closer to wrong sense", color=OUTCOME["wrong"])
    ax.text(0.98, len(layers) - 0.63, "closer to intended sense", color=OUTCOME["correct"], ha="right")
    ax.legend(loc="lower center", ncol=2)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    _save_svg_and_png(fig, output_path)


def _read_curve(path: Path, value_column: str):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.array([int(row["Layer"]) for row in rows]),
        np.array([float(row[value_column]) for row in rows]),
    )


def _read_h1_best_layer(path: Path, word: str):
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["word"] == word:
                return int(row["best_layer_M"])
    raise KeyError(word)


def _make_curve_figure(
    results_dir,
    safe_model,
    word,
    layers,
    model_label,
    output_path,
):
    h1_dir = results_dir / "study" / "H1" / safe_model
    adequacy_layers, fractions = _read_curve(h1_dir / f"h1_{word}.csv", "FractionAdequate")
    gdv_layers, gdv_values = _read_curve(
        results_dir / f"{safe_model}_gdv" / f"gdv_values_{word}.csv",
        "GDV",
    )
    adequacy_best = _read_h1_best_layer(h1_dir / "h1_summary.csv", word)
    contextual = gdv_layers != 0
    gdv_best = int(gdv_layers[contextual][np.argmin(gdv_values[contextual])])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), constrained_layout=True)
    adequacy_ax, gdv_ax = axes
    adequacy_ax.plot(adequacy_layers, fractions, color=METHOD["adequacy"], linewidth=2.1)
    adequacy_ax.scatter(adequacy_layers, fractions, color=METHOD["adequacy"], s=18)
    adequacy_value = fractions[np.where(adequacy_layers == adequacy_best)[0][0]]
    adequacy_ax.scatter(
        adequacy_best,
        adequacy_value,
        s=95,
        facecolor="white",
        edgecolor=METHOD["adequacy"],
        linewidth=2.2,
        zorder=5,
    )
    adequacy_ax.annotate(
        f"best L{adequacy_best}",
        (adequacy_best, adequacy_value),
        xytext=(7, 10),
        textcoords="offset points",
        fontweight="bold",
    )
    adequacy_ax.set_ylim(0.35, 1.03)
    adequacy_ax.set_ylabel("Leave-one-out adequate fraction · higher is better")
    adequacy_ax.set_title(r"Sentence decisions ($M_l$)", loc="left", fontweight="bold")

    gdv_ax.plot(gdv_layers, gdv_values, color=METHOD["gdv"], linewidth=2.1)
    gdv_ax.scatter(gdv_layers, gdv_values, color=METHOD["gdv"], s=18)
    gdv_value = gdv_values[np.where(gdv_layers == gdv_best)[0][0]]
    gdv_ax.scatter(
        gdv_best,
        gdv_value,
        s=95,
        facecolor="white",
        edgecolor=METHOD["gdv"],
        linewidth=2.2,
        zorder=5,
    )
    gdv_ax.annotate(
        f"best L{gdv_best}",
        (gdv_best, gdv_value),
        xytext=(7, -16),
        textcoords="offset points",
        fontweight="bold",
        color=METHOD["gdv"],
    )
    gdv_ax.set_ylabel("Corrected GDV · more negative is stronger")
    gdv_ax.set_title("Global sense geometry (GDV)", loc="left", fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Layer")
        ax.axvline(max(adequacy_layers), color=METHOD["final"], linestyle=":", linewidth=1.4)
        for layer in layers:
            ax.axvline(layer, color=GRID, linestyle="--", linewidth=1.0)
    fig.suptitle(
        f"Layer-wise evidence · {model_label} · {word}",
        fontsize=14,
        fontweight="bold",
    )
    _save_svg_and_png(fig, output_path)


def make_figures(results_dir: Path, safe_model: str, word: str, layers, pca_output: Path):
    apply_report_style()
    layer_data = {}
    for layer in layers:
        X, labels = _load_layer(results_dir, safe_model, word, layer)
        raw, normalized = leave_one_out_adequacy_margins(X, labels)
        layer_data[layer] = {
            "X": X,
            "labels": labels,
            "normalized": normalized,
            "fraction": float((raw > 0).mean()),
            "gdv": compute_gdv(X, labels),
        }

    output_dir = pca_output.parent
    model_label = safe_model.replace("_", "/")
    _make_pca_figure(layer_data, layers, model_label, word, pca_output)
    _make_score_figure(
        layer_data,
        layers,
        model_label,
        word,
        output_dir / "h2_decision_scores.svg",
    )
    _make_curve_figure(
        results_dir,
        safe_model,
        word,
        layers,
        model_label,
        output_dir / "h2_layer_curves.svg",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model", default="Qwen_Qwen2.5-7B")
    parser.add_argument("--word", default="bat")
    parser.add_argument("--layers", nargs=2, type=int, default=[3, 26])
    parser.add_argument(
        "--output",
        default="results/study/figures/h2_gdv_geometric_example.svg",
        help="PCA output path; PNG is also written, along with two companion figures.",
    )
    args = parser.parse_args()
    make_figures(
        Path(args.results_dir),
        args.model,
        args.word,
        args.layers,
        Path(args.output),
    )


if __name__ == "__main__":
    main()
