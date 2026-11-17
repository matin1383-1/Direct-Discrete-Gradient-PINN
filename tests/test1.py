"""20,000-step comparison of all five methods on the 1-D Poisson problem.

The experiment uses the same seed, nodes, network architecture, and manufactured
reference for every method.  It records every optimizer step so reported best
relative errors are not limited to sparse logging checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from train import TrainingConfig, run_training


METHODS = ("gpinn", "dt3", "dt4", "dt5", "direct")
COLORS = {
    "gpinn": "#9467bd",
    "dt3": "#8c564b",
    "dt4": "#e377c2",
    "dt5": "#7f7f7f",
    "direct": "#1f77b4",
}


def _log_ema(values, smoothing: float = 0.02) -> np.ndarray:
    """Exponential trend in log space, appropriate for positive error/loss curves."""
    values = np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)
    logs = np.log10(values)
    trend = np.empty_like(logs)
    trend[0] = logs[0]
    for index in range(1, len(logs)):
        trend[index] = smoothing * logs[index] + (1.0 - smoothing) * trend[index - 1]
    return 10.0**trend


def _method_summary(method: str, records: list[dict]) -> dict:
    l1 = np.asarray([row["relative_l1_error"] for row in records], dtype=float)
    l2 = np.asarray([row["relative_l2_error"] for row in records], dtype=float)
    best_l1_index = int(np.argmin(l1))
    best_l2_index = int(np.argmin(l2))
    final = records[-1]
    return {
        "method": method,
        "steps": int(final["optimizer_step"]),
        "final_loss": float(final["total_loss"]),
        "final_relative_l1": float(final["relative_l1_error"]),
        "final_relative_l2": float(final["relative_l2_error"]),
        "best_relative_l1": float(l1[best_l1_index]),
        "best_relative_l1_step": int(records[best_l1_index]["optimizer_step"]),
        "best_relative_l2": float(l2[best_l2_index]),
        "best_relative_l2_step": int(records[best_l2_index]["optimizer_step"]),
        "training_update_seconds": float(final["training_update_time"]),
        "operator_construction_seconds": float(final["operator_construction_time"]),
        "total_process_seconds": float(final["total_process_time"]),
    }


def _plot_loss(histories: dict[str, list[dict]], destination: Path) -> None:
    plt.figure(figsize=(8.2, 5.2))
    for method, records in histories.items():
        steps = [row["optimizer_step"] for row in records]
        loss = [row["total_loss"] for row in records]
        plt.semilogy(steps, loss, color=COLORS[method], linewidth=0.35, alpha=0.16)
        plt.semilogy(
            steps,
            _log_ema(loss),
            color=COLORS[method],
            label=f"{method.capitalize()} trend",
            linewidth=1.6,
        )
    plt.xlabel("Adam optimizer step")
    plt.ylabel("Total training loss")
    final_step = max(records[-1]["optimizer_step"] for records in histories.values())
    plt.title(f"Poisson 1D: {final_step:,}-step loss comparison")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def _plot_errors(histories: dict[str, list[dict]], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    for method, records in histories.items():
        steps = [row["optimizer_step"] for row in records]
        for axis, field in zip(axes, ("relative_l1_error", "relative_l2_error")):
            values = [row[field] for row in records]
            axis.semilogy(steps, values, color=COLORS[method], linewidth=0.3, alpha=0.12)
            axis.semilogy(
                steps,
                _log_ema(values),
                color=COLORS[method],
                label=f"{method.capitalize()} trend",
                linewidth=1.5,
            )
    axes[0].set_title("Relative L1 error")
    axes[1].set_title("Relative L2 error")
    for axis in axes:
        axis.set_xlabel("Adam optimizer step")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Relative error")
    figure.suptitle("Poisson 1D solution-error convergence")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def run_experiment(
    *,
    steps: int = 20_000,
    seed: int = 200,
    device: str = "cpu",
    dtype: str = "float32",
    output_dir: str = "outputs/test1",
) -> list[dict]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # Each method builds only its own operators on identical deterministic nodes.
    histories: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    for method in METHODS:
        print(f"\n=== {method.upper()}: {steps} steps, seed {seed} ===", flush=True)
        config = TrainingConfig(
            method=method,
            problem="poisson1d",
            steps=steps,
            seed=seed,
            device=device,
            dtype=dtype,
            output_dir=str(destination / "runs"),
            log_every=1,
        )
        _, records, _ = run_training(
            config, save_outputs=True, verbose=False
        )
        histories[method] = records
        summary = _method_summary(method, records)
        run_summary = json.loads((destination / "runs" / f"poisson1d_{method}_seed{seed}" / "run_summary.json").read_text(encoding="utf-8"))
        summary["total_process_seconds"] = run_summary["total_process_time"]
        summary["metrics_schema_version"] = 2
        summaries.append(summary)
        print(
            f"final: loss={summary['final_loss']:.6e}, "
            f"rel-L1={summary['final_relative_l1']:.6e}, "
            f"rel-L2={summary['final_relative_l2']:.6e}"
        )
        print(
            f"best:  rel-L1={summary['best_relative_l1']:.6e} "
            f"at step {summary['best_relative_l1_step']}, "
            f"rel-L2={summary['best_relative_l2']:.6e} "
            f"at step {summary['best_relative_l2_step']}"
        )

    _plot_loss(histories, destination / "loss_comparison.png")
    _plot_errors(histories, destination / "relative_error_comparison.png")
    (destination / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nComparison artifacts written to {destination.resolve()}")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", default="outputs/test1")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(
        steps=arguments.steps,
        seed=arguments.seed,
        device=arguments.device,
        dtype=arguments.dtype,
        output_dir=arguments.output_dir,
    )
