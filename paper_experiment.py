"""Reproduce the seven final configurations from the DDG-PINN paper.

The protocol follows Section IV of "Direct Discrete Training of
Gradient-Enhanced Physics-Informed Neural Networks": three seeds, 20,000 Adam
updates, fixed validation samples, and a separate 1,001-point final test grid.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import TrainingConfig, relative_errors, run_training


@dataclass(frozen=True)
class PaperConfiguration:
    key: str
    label: str
    method: str
    lambda_weight: float


PAPER_CONFIGURATIONS = (
    PaperConfiguration("pinn", "PINN", "pinn", 0.0),
    PaperConfiguration("gpinn_w0.1", "gPINN, lambda=0.1", "gpinn", 0.1),
    PaperConfiguration("ddg_w0.01", "DDG-PINN, omega=0.01", "ddg", 0.01),
    PaperConfiguration("ddg_w1", "DDG-PINN, omega=1", "ddg", 1.0),
    PaperConfiguration("ddg_w0.1", "DDG-PINN, omega=0.1", "ddg", 0.1),
    PaperConfiguration("dt3", "DT-PINN order 3", "dt3", 0.0),
    PaperConfiguration("dt4", "DT-PINN order 4", "dt4", 0.0),
)

COLORS = {
    "pinn": "#4d4d4d",
    "gpinn_w0.1": "#9467bd",
    "ddg_w0.01": "#009e73",
    "ddg_w1": "#e69f00",
    "ddg_w0.1": "#0072b2",
    "dt3": "#cc79a7",
    "dt4": "#8c564b",
}


def paper_validation_points() -> np.ndarray:
    """The fixed 1,000-point uniform random validation set (NumPy seed 321)."""
    random_state = np.random.RandomState(321)
    return np.sort(random_state.uniform(0.0, pi, size=1_000))[:, None]


def paper_test_points() -> np.ndarray:
    """The separate uniform 1,001-point final test grid."""
    return np.linspace(0.0, pi, 1_001)[:, None]


def _mean_and_sample_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    sample_std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return float(np.mean(array)), sample_std


def _plot_validation(histories: dict[str, list[list[dict]]], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for configuration in PAPER_CONFIGURATIONS:
        runs = histories[configuration.key]
        for records in runs:
            axis.semilogy(
                [row["optimizer_step"] for row in records],
                [row["relative_l2_error"] for row in records],
                color=COLORS[configuration.key],
                alpha=0.15,
                linewidth=0.8,
            )
        steps = np.asarray([row["optimizer_step"] for row in runs[0]])
        values = np.asarray(
            [[row["relative_l2_error"] for row in records] for records in runs]
        )
        axis.semilogy(
            steps,
            np.mean(values, axis=0),
            color=COLORS[configuration.key],
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            label=configuration.label,
        )
    axis.set_xlabel("Adam optimizer steps")
    axis.set_ylabel(r"Validation relative $L^2$ error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def run_paper_experiment(
    *,
    steps: int = 20_000,
    seeds: tuple[int, ...] = (200, 201, 202),
    device: str = "cpu",
    output_dir: str = "outputs/paper_reproduction",
) -> list[dict]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    validation_points = paper_validation_points()
    test_points = paper_test_points()
    log_every = min(1_000, steps)
    histories: dict[str, list[list[dict]]] = {
        configuration.key: [] for configuration in PAPER_CONFIGURATIONS
    }
    individual_results: list[dict] = []

    protocol = {
        "paper_title": "Direct Discrete Training of Gradient-Enhanced Physics-Informed Neural Networks",
        "steps": steps,
        "seeds": list(seeds),
        "device": device,
        "dtype": "float32",
        "learning_rate": 1.0e-3,
        "validation_seed": 321,
        "validation_points": 1_000,
        "validation_interval": log_every,
        "test_points": 1_001,
        "configurations": [asdict(value) for value in PAPER_CONFIGURATIONS],
    }
    (destination / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    for configuration in PAPER_CONFIGURATIONS:
        for seed in seeds:
            print(f"\n=== {configuration.label}; seed={seed}; steps={steps} ===", flush=True)
            training = TrainingConfig(
                method=configuration.method,
                problem="poisson1d",
                steps=steps,
                seed=seed,
                device=device,
                dtype="float32",
                learning_rate=1.0e-3,
                lambda_weight=configuration.lambda_weight,
                output_dir=str(destination / configuration.key),
                log_every=log_every,
            )
            model, records, problem = run_training(
                training,
                save_outputs=True,
                verbose=False,
                evaluation_points=validation_points,
            )
            test_l1, test_l2 = relative_errors(problem, model, coordinates=test_points)
            histories[configuration.key].append(records)
            result = {
                "configuration": configuration.key,
                "label": configuration.label,
                "method": configuration.method,
                "lambda_weight": configuration.lambda_weight,
                "seed": seed,
                "optimizer_steps": steps,
                "test_relative_l1": test_l1,
                "test_relative_l2": test_l2,
                "training_seconds": records[-1]["training_update_time"],
                "operator_construction_seconds": records[-1]["operator_construction_time"],
            }
            individual_results.append(result)
            print(
                f"test relative L2={test_l2:.6e}; "
                f"training={result['training_seconds']:.2f}s"
            )

    summary: list[dict] = []
    for configuration in PAPER_CONFIGURATIONS:
        rows = [
            row for row in individual_results if row["configuration"] == configuration.key
        ]
        error_mean, error_std = _mean_and_sample_std(
            [float(row["test_relative_l2"]) for row in rows]
        )
        time_mean, time_std = _mean_and_sample_std(
            [float(row["training_seconds"]) for row in rows]
        )
        summary.append(
            {
                "configuration": configuration.key,
                "method": configuration.label,
                "test_relative_l2_mean": error_mean,
                "test_relative_l2_sample_std": error_std,
                "training_seconds_mean": time_mean,
                "training_seconds_sample_std": time_std,
                "seeds": len(rows),
            }
        )

    (destination / "individual_results.json").write_text(
        json.dumps(individual_results, indent=2), encoding="utf-8"
    )
    (destination / "paper_table.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (destination / "paper_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    _plot_validation(histories, destination / "validation_relative_l2.png")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[200, 201, 202])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/paper_reproduction")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one update and only the first seed for a pipeline check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.steps = 1
        args.seeds = args.seeds[:1]
    run_paper_experiment(
        steps=args.steps,
        seeds=tuple(args.seeds),
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
