"""Command-line and programmatic training pipeline for all seven methods."""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from methods import (
    METHOD_NAMES,
    compute_loss,
)
from networks import make_network
from problems import (
    LAMBDA_DEFAULTS,
    LEARNING_RATES,
    PROBLEM_NAMES,
    exact_solution,
    evaluation_coordinates,
    make_problem,
)


@dataclass
class TrainingConfig:
    method: str
    problem: str
    steps: int = 20_000
    seed: int = 200
    device: str = "cpu"
    dtype: str = "float32"
    learning_rate: float | None = None
    lambda_weight: float | None = None
    output_dir: str = "outputs"
    log_every: int = 100

    def resolved(self) -> "TrainingConfig":
        if self.method not in METHOD_NAMES:
            raise ValueError(f"method must be one of {METHOD_NAMES}")
        if self.problem not in PROBLEM_NAMES:
            raise ValueError(f"problem must be one of {PROBLEM_NAMES}")
        if self.steps < 1 or self.log_every < 1:
            raise ValueError("steps and log_every must be positive")
        # gPINN and DT-PINN use the documented deterministic Adam default
        # unless explicitly overridden on the CLI.
        if self.learning_rate is None:
            self.learning_rate = LEARNING_RATES[self.problem].get(self.method, 1.0e-3)
        self.lambda_weight = (
            LAMBDA_DEFAULTS[self.problem] if self.lambda_weight is None else self.lambda_weight
        )
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.lambda_weight) or self.lambda_weight < 0:
            raise ValueError("lambda_weight must be finite and nonnegative")
        return self


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def relative_errors(problem, model) -> tuple[float | None, float | None]:
    exact = exact_solution(problem.name)
    if exact is None:
        return None, None
    with torch.no_grad():
        coordinates = problem.tensor(evaluation_coordinates(problem.name))
        prediction = torch.cat([model(batch) for batch in coordinates.split(4096)])
        reference = exact(coordinates)
        difference = prediction - reference
        relative_l1 = torch.sum(torch.abs(difference)) / torch.sum(torch.abs(reference))
        relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
        return float(relative_l1), float(relative_l2)


def relative_l2_error(problem, model) -> float | None:
    """Backward-compatible convenience wrapper used by external callers."""
    return relative_errors(problem, model)[1]


def _save_plot(records: list[dict], destination: Path) -> None:
    plt.figure(figsize=(7.0, 4.5))
    plt.semilogy([r["optimizer_step"] for r in records], [r["total_loss"] for r in records])
    plt.xlabel("optimizer step")
    plt.ylabel("total loss")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination, dpi=150)
    plt.close()


def run_training(
    config: TrainingConfig,
    *,
    problem=None,
    save_outputs: bool = True,
    verbose: bool = True,
):
    config = config.resolved()
    set_reproducible_seed(config.seed)
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    process_started = perf_counter()
    operators_reused = problem is not None
    if problem is None:
        problem = make_problem(config.problem, device=config.device, dtype=dtype, method=config.method)
    else:
        if problem.name != config.problem:
            raise ValueError("provided problem does not match configuration")
        problem.to(config.device, dtype)
    def synchronize():
        if problem.device.type == "cuda":
            torch.cuda.synchronize(problem.device)

    operator_seconds = 0.0 if operators_reused else problem.operator_seconds
    model = make_network(problem, device=config.device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    run_dir = Path(config.output_dir) / f"{config.problem}_{config.method}_seed{config.seed}"
    if save_outputs:
        # A repeated seed must not silently destroy a previous experiment.
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}; choose a new --output-dir")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
        )
        diagnostics = {name: asdict(value) for name, value in problem.diagnostics.items()}
        (run_dir / "operator_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )

    records: list[dict] = []
    best_l2 = float("inf")
    best_l2_state = None
    best_l2_record = None
    training_seconds = 0.0
    evaluation_seconds = 0.0
    for step in range(config.steps):
        synchronize()
        update_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        breakdown = compute_loss(
            problem,
            model,
            method=config.method,
            lambda_weight=config.lambda_weight,
        )
        if not torch.isfinite(breakdown.total):
            raise FloatingPointError(f"non-finite loss at optimizer step {step}")
        breakdown.total.backward()
        for parameter in model.parameters():
            if parameter.grad is None or not torch.all(torch.isfinite(parameter.grad)):
                raise FloatingPointError(f"missing or non-finite parameter gradient at step {step}")
        optimizer.step()
        synchronize()
        training_seconds += perf_counter() - update_started

        should_log = step == 0 or (step + 1) % config.log_every == 0 or step + 1 == config.steps
        if should_log:
            evaluate_started = perf_counter()
            # Re-evaluate after the update from the current model state.
            # Coordinate AD requires grad mode; this pass never calls backward.
            measured = compute_loss(
                problem, model, method=config.method,
                lambda_weight=config.lambda_weight,
            )
            if not torch.isfinite(measured.total):
                raise FloatingPointError(f"non-finite post-update loss at step {step + 1}")
            measured_scalars = measured.scalars()
            del measured
            relative_l1, relative_l2 = relative_errors(problem, model)
            synchronize()
            evaluation_seconds += perf_counter() - evaluate_started
            record = {
                "method": config.method,
                "problem": config.problem,
                "seed": config.seed,
                "optimizer_step": step + 1,
                **measured_scalars,
                "metrics_schema_version": 2,
                "relative_l1_error": relative_l1,
                "relative_l2_error": relative_l2,
                "training_update_time": training_seconds,
                "operator_construction_time": operator_seconds,
                "operators_reused": operators_reused,
                "evaluation_time": evaluation_seconds,
                "total_process_time": perf_counter() - process_started,
            }
            records.append(record)
            if relative_l2 is not None and relative_l2 < best_l2:
                best_l2 = relative_l2
                best_l2_record = dict(record)
                best_l2_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            if verbose:
                error_text = (
                    "n/a" if record["relative_l2_error"] is None else f"{record['relative_l2_error']:.3e}"
                )
                print(
                    f"[{config.method}/{config.problem}] step={step + 1:6d} "
                    f"loss={record['total_loss']:.6e} rel_l2={error_text}"
                )

    if save_outputs:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
            },
            run_dir / "checkpoint_final.pt",
        )
        if best_l2_state is not None:
            torch.save(
                {
                    "model_state_dict": best_l2_state,
                    "config": asdict(config),
                    "best_record": best_l2_record,
                },
                run_dir / "checkpoint_best_observed_reference_l2.pt",
            )
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        with (run_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        _save_plot(records, run_dir / "training_loss.png")
        synchronize()
        (run_dir / "run_summary.json").write_text(json.dumps({
            "metrics_schema_version": 2,
            "evaluation": "independent midpoint samples; discrete relative norms",
            "checkpoint_selection": "minimum reference L2 over logged post-update states; retrospective only",
            "log_every": config.log_every,
            "best_observed_reference_l2": None if best_l2_record is None else best_l2_record,
            "final_record": records[-1],
            "operator_construction_time": operator_seconds,
            "operators_reused": operators_reused,
            "training_update_time": training_seconds,
            "evaluation_time": evaluation_seconds,
            "total_process_time": perf_counter() - process_started,
            "timing_scope": "through checkpoint, history and plot writes; excludes this summary write",
        }, indent=2), encoding="utf-8")
    return model, records, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_NAMES, required=True)
    parser.add_argument("--problem", choices=PROBLEM_NAMES, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lambda-weight", type=float)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--smoke-test", action="store_true", help="cap the requested run at 10 optimizer steps"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.steps = min(args.steps, 10)
        args.log_every = min(args.log_every, args.steps)
    config = TrainingConfig(
        method=args.method,
        problem=args.problem,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        learning_rate=args.learning_rate,
        lambda_weight=args.lambda_weight,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )
    run_training(config)


if __name__ == "__main__":
    main()
