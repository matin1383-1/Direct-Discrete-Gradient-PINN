# Standalone gPINN, DT-PINN, and Hybrid Methods

This directory is a from-scratch implementation of seven independently selectable
physics-informed configurations sharing one problem and network backbone:

- `gpinn`: the continuous PDE residual and all of its coordinate derivatives are
  computed by PyTorch AD over the complete eligible collocation set.
- `dt3`, `dt4`, `dt5`: basic DT-PINNs whose spatial PDE derivatives use independent
  PHS RBF-FD operators with nominal approximation order 3, 4, or 5. They minimize
  the discrete PDE residual without a residual-gradient penalty. Time derivatives
  remain AD.

- `direct`: the PDE residual and the analytically derived residual-gradient terms
  are discretized directly with sparse PHS RBF-FD matrices.
- `adaptive`: the same direct loss plus a genuine pointwise-AD residual and
  residual-gradient correction at points selected by a p=3/p=4 Laplacian
  disagreement indicator and deterministic rotating coverage.
- `fixed`: exactly the same hybrid loss as `adaptive`, but correction points are
  chosen once at evenly spaced candidate indices and never changed.

The implementation does not differentiate `D @ residual` as a substitute for the
specified analytical residual gradients. For time-dependent PDEs, sparse matrices
act only along the spatial axis; `u_t`, `u_tt`, and the correction derivatives use
PyTorch automatic differentiation.

## Files

- `rbf_fd.py`: PHS RBF-FD weights, polynomial augmentation, nearest-neighbor
  stencils, direct mixed derivatives, diagnostics, and SciPy-to-PyTorch sparse
  conversion.
- `problems.py`: the five PDEs, manufactured data, deterministic collocation
  clouds, and operator construction.
- `networks.py`: tanh MLPs, Glorot initialization, and hard output transforms.
- `methods.py`: all seven loss paths, AD corrections, DT order routing,
  disagreement calculation, and correction selectors.
- `train.py`: deterministic full-batch Adam training, CLI, timings, checkpoints,
  CSV/JSONL logs, and loss plots.
- `tests/test_methods.py`: numerical derivative, residual, constraint, selection,
  differentiability, end-to-end, disk, Robin, gPINN, and DT-order tests.
- `tests/test1.py`: the long Poisson-1D comparison across all seven configurations.

## Installation and tests

Python 3.11 or newer is required. `requirements.txt` pins the exact versions
used for the reported verification (PyTorch 2.9.0, NumPy 2.3.4, SciPy 1.16.2,
and Matplotlib 3.10.7).

```bash
cd standalone_hybrid_pinn
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

The test suite checks smooth first/second/third and mixed RBF-FD derivatives,
polynomial reproduction, explicit Burgers residual-gradient formulas against an
independent analytic reference, hard constraints, both selectors, gradient flow,
all seven training paths on steady and time-dependent PDEs, and the disk/Robin
path. The DT tests verify distinct order-3/4/5 matrices, and the gPINN test checks
the AD residual and its derivative against the manufactured exact solution.

## Training

```bash
python train.py --method direct --problem poisson1d
python train.py --method gpinn --problem poisson1d
python train.py --method dt3 --problem poisson2d
python train.py --method dt4 --problem burgers
python train.py --method dt5 --problem heat
python train.py --method adaptive --problem burgers
python train.py --method fixed --problem heat
python train.py --method adaptive --problem poisson2d --steps 500 --seed 200
python train.py --method direct --problem poisson1d --smoke-test
```

For the dedicated equal-seed, 20,000-step Poisson-1D comparison of all seven
configurations with per-step relative L1/L2 tracking and combined plots:

```bash
python tests/test1.py
```

Results are written to `outputs/test1/`, including final and best observed errors, combined
loss/error figures, every method's history, and best-L2/final checkpoints.

Supported problem names are `poisson1d`, `poisson2d`, `burgers`, `allen_cahn`, and
`heat`. Method choices are `gpinn`, `dt3`, `dt4`, `dt5`, `direct`, `adaptive`,
and `fixed`. The CLI also accepts `--device`, `--dtype`, `--learning-rate`,
`--lambda-weight`, `--beta`, `--correction-count`, `--adapt-every`, and
`--output-dir`. The original direct/adaptive/fixed defaults remain frozen. Because
no new learning rates were specified for the added baselines, gPINN and all three
DT variants use Adam at `1e-3` by default; `--learning-rate` overrides it. The
existing problem-specific lambda is used by gPINN, while DT-PINN has no gradient
penalty. Seeds 200, 201, and 202 can be used as independent deterministic runs.

Each run writes its resolved configuration, per-operator diagnostics, final model
and optimizer checkpoint, CSV and JSONL histories, and `training_loss.png` below
`outputs/<problem>_<method>_seed<seed>/`. Operator construction, optimizer update,
adaptive selection, and total process times are logged separately. Relative L1 and
L2 errors are reported only for Poisson 1D, Poisson 2D, and manufactured heat; they are
deliberately reported as unavailable for Burgers and Allen-Cahn because no
independent reference arrays are supplied.

## RBF-FD details and assumptions

For derivative order `s`, polynomial degree is `q=p+s-1`, the nominal stencil is
`2*binomial(q+d,d)+1`, and local coordinates are centered and scaled by the
farthest stencil radius. The PHS exponent is the smallest odd integer greater than
or equal to `q`, clipped to `[5,11]`. This deterministic rule is an assumption
because the requested exponent mapping was unspecified. Weights are rescaled by
the radius to power `-s`.

Near a Cartesian boundary, the nominal nearest set can be non-unisolvent even when
it contains enough points. In that explicit failure case, the code adds the next
nearest points one at a time until the polynomial block has full rank. It never
silently substitutes a pseudoinverse. Duplicate clouds, remaining rank failures,
non-finite systems, and condition estimates above the documented limit raise an
exception. Diagnostics report the maximum stencil actually required, polynomial
reproduction error, local condition statistics, sparsity, and construction time.

The default condition-number ceiling is `1e12`. Each local solve must have
normalized backward error at most `1e-12` and normalized polynomial reproduction
error at most `1e-7`. The assembled float64 operator must also reproduce its
polynomial basis within `1e-7`. Failures raise an exception before training.
These checks apply to construction precision; conversion to float32 can still
introduce cancellation, especially in third derivatives. Use `--dtype float64`
for accuracy investigations.

Poisson 1D uses 129 uniform nodes and Poisson 2D uses a 21-by-21 Cartesian cloud,
because their counts were not specified. The unavailable 828-node heat cloud is
replaced by a deterministic 10-ring polar cloud (441 physical points) and two
outward ghost rings at radii 1.04 and 1.08 (601 support points total). PDE rows are
interior physical points; ghost values only support spatial stencils. Robin normal
derivatives are evaluated with the same discrete `D_x` and `D_y` operators for
DT/direct/hybrid methods. The AD-only gPINN evaluates the Robin normal derivative
with AD. For DT-PINN, both the PDE and Robin first/second spatial derivatives use
the selected order's independently constructed operators.

These choices preserve the specified algorithms and allow their numerical effects
to be isolated. They do not claim exact reproduction of the unavailable original
point cloud, unspecified collocation counts, optimizer trajectory, or research
repository implementation.

## Evaluation, timing, and checkpoint interpretation

Metrics schema version 2 logs losses and relative errors from the same model
**after** the stated optimizer update. Logging re-evaluates the loss using the
current correction subset; it does not refresh that subset or update parameters.
AD fields are named `ad_residual_loss` and `ad_gradient_loss`: they are the main
gPINN terms or the hybrid correction terms, depending on the selected method.

Relative errors use independent deterministic samples: 2,048 midpoint nodes for
Poisson 1D, an 80-by-80 midpoint grid for Poisson 2D, and 24 equal-area radial
levels by 64 angles by 20 midpoint times for heat. These are discrete relative
L1/L2 norms, not exact continuum integrals. Reference data are used only for
evaluation and retrospective checkpoint selection.

`checkpoint_final.pt` contains the final model and optimizer state.
`checkpoint_best_observed_reference_l2.pt` contains the model with the lowest
reference L2 at a logged step. This is a retrospective diagnostic, not an early
stopping rule or a claim of independently validated model selection. The default
logging interval is 100; `tests/test1.py` evaluates every step. Best-L1 values in
the comparison summary are descriptive; a separate best-L1 checkpoint is not saved.

Normal CLI runs construct only the selected method's operators; gPINN constructs
none. `make_problem(..., method=None)` remains available as an all-operator bank
for numerical audits. Supplying an existing `problem` to `run_training` is an
explicit reuse path: its operators must support the method, and current-run
construction time is reported as zero with `operators_reused=true`.

GPU timing synchronizes at optimizer and adaptation boundaries. Training update,
operator construction, adaptation, and logging evaluation times are separated.
History rows contain elapsed time at observation. `run_summary.json` reports
elapsed time through checkpoint, history, and plot writes, excluding only the
summary's own write. The comparison driver uses that summary for run totals.
For fair comparisons, inspect solution errors as well as loss: different methods
minimize different objectives and use different default learning rates.

## Repository and local results

Source, documentation, tests, and the CPU CI workflow belong in Git. Generated
results and checkpoints are excluded by `.gitignore`. Historical results are
preserved locally under `artifacts/legacy/`; their metrics use the old collocation
grid and pre-update loss convention and must not be pooled with schema version 2.
New runs refuse to overwrite a nonempty run directory; use a new `--output-dir`.

Run the verification suite before publishing. The GitHub workflow runs the same
suite on Python 3.11/CPU; GPU behavior requires a separate CUDA-enabled machine.
No remote repository or license is assumed by this project.
