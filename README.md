# Direct Discrete-Gradient Training of Physics-Informed Neural Networks

A standalone PyTorch implementation for comparing five PINN training methods on
the same PDE definitions, collocation clouds, neural-network family, optimizer,
and reporting pipeline.

The repository studies how replacing automatic spatial differentiation with
sparse polyharmonic-spline radial-basis-function finite differences (PHS RBF-FD)
changes PINN training cost and accuracy.

## Implemented methods

Each problem uses a fully connected tanh network. The methods differ in how they
evaluate the PDE residual and, when applicable, its coordinate derivatives.

| CLI method | PDE derivatives | Residual-gradient term |
| --- | --- | --- |
| `gpinn` | PyTorch automatic differentiation (AD) | AD at every PDE point |
| `dt3` | Order-3 PHS RBF-FD in space; AD in time | None |
| `dt4` | Order-4 PHS RBF-FD in space; AD in time | None |
| `dt5` | Order-5 PHS RBF-FD in space; AD in time | None |
| `direct` | Order-3 PHS RBF-FD in space; AD in time | Analytically derived and directly discretized |

The `direct` method discretizes the analytical expressions for the coordinate
derivatives of the residual. It does not approximate them by applying a
differentiation matrix to an already evaluated residual.

The losses are:

\[
L_{\mathrm{gPINN}} = \operatorname{MSE}(r)
+ \lambda \sum_j \operatorname{MSE}(\partial_j r)
+ L_{\mathrm{constraints}},
\]

\[
L_{\mathrm{DT}p} = \operatorname{MSE}(r_p)
+ L_{\mathrm{constraints}}, \qquad p \in \{3,4,5\},
\]

\[
L_{\mathrm{direct}} = \operatorname{MSE}(r_3)
+ \lambda \sum_j \operatorname{MSE}(g_{3,j})
+ L_{\mathrm{constraints}},
\]

where `r_p` is the residual evaluated with order-`p` spatial RBF-FD operators
and `g_{3,j}` is a directly discretized analytical residual derivative.

## Supported PDEs

| CLI problem | Domain | Equation type | Independent reference metrics |
| --- | --- | --- | --- |
| `poisson1d` | `[0, pi]` | Steady Poisson | Relative L1 and L2 |
| `poisson2d` | `[0, 1]^2` | Steady Poisson | Relative L1 and L2 |
| `burgers` | `[-1, 1] x [0, 1]` | Time-dependent viscous Burgers | Not reported |
| `allen_cahn` | `[-1, 1] x [0, 1]` | Time-dependent Allen-Cahn | Not reported |
| `heat` | Unit disk x `[0, 1]` | Manufactured heat equation with Robin data | Relative L1 and L2 |

Poisson, Burgers, and Allen-Cahn enforce their prescribed boundary and initial
conditions through exact output transformations. The heat problem includes its
initial condition and Robin boundary condition as loss terms. For heat, gPINN
uses AD normal derivatives while the discrete methods use their corresponding
RBF-FD first-derivative operators.

## Installation

Python 3.11 or newer is recommended.

```bash
git clone <repository-url>
cd standalone_hybrid_pinn
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Running an experiment

Use `--smoke-test` for a run of at most ten optimizer updates:

```bash
python train.py --method direct --problem poisson1d --smoke-test
```

Run a complete experiment:

```bash
python train.py --method direct --problem poisson1d --steps 20000 --seed 200
```

Additional examples:

```bash
# Fully AD gradient-enhanced PINN
python train.py --method gpinn --problem poisson1d

# Discrete-training PINNs with different RBF-FD approximation orders
python train.py --method dt3 --problem poisson2d
python train.py --method dt4 --problem burgers
python train.py --method dt5 --problem heat --dtype float64

# Directly discretized residual-gradient loss
python train.py --method direct --problem allen_cahn --device cuda
```

The five valid method names are `gpinn`, `dt3`, `dt4`, `dt5`, and `direct`.
The five valid problem names are `poisson1d`, `poisson2d`, `burgers`,
`allen_cahn`, and `heat`.

### Command-line options

Run `python train.py --help` for the authoritative CLI. The available options
are:

| Option | Default | Description |
| --- | --- | --- |
| `--method` | Required | Training method |
| `--problem` | Required | PDE problem |
| `--steps` | `20000` | Number of Adam updates |
| `--seed` | `200` | Python, NumPy, and PyTorch seed |
| `--device` | `cpu` | PyTorch device, such as `cpu` or `cuda` |
| `--dtype` | `float32` | Training type: `float32` or `float64` |
| `--learning-rate` | Method/problem dependent | Override the Adam learning rate |
| `--lambda-weight` | Problem dependent | Weight of the residual-gradient term |
| `--output-dir` | `outputs` | Parent directory for run artifacts |
| `--log-every` | `100` | Metric logging interval in optimizer updates |
| `--smoke-test` | Disabled | Limit the requested run to at most ten updates |

gPINN and all DT-PINN variants default to a learning rate of `1e-3`. The direct
method defaults are:

| Problem | Direct learning rate | Lambda |
| --- | ---: | ---: |
| `poisson1d` | `1e-3` | `0.1` |
| `poisson2d` | `3e-4` | `0.01` |
| `burgers` | `3e-4` | `1e-7` |
| `allen_cahn` | `1e-3` | `1e-5` |
| `heat` | `3e-4` | `0.01` |

The lambda defaults apply to gPINN and direct training. DT-PINN has no
residual-gradient term, so lambda does not affect its loss.

## Five-method comparison

`tests/test1.py` runs all five methods on Poisson 1D with identical nodes,
network architecture, seed, evaluation points, and optimizer-step count. It logs
every optimizer update.

```bash
# Default: 20,000 updates per method
python tests/test1.py

# Short pipeline check
python tests/test1.py --steps 10 --output-dir outputs/comparison_smoke
```

The comparison writes `summary.csv`, `summary.json`, combined loss and error
plots, and a normal run directory for each method.

## Run artifacts and metrics

A normal run is written to:

```text
outputs/<problem>_<method>_seed<seed>/
|-- config.json
|-- operator_diagnostics.json
|-- history.csv
|-- history.jsonl
|-- training_loss.png
|-- checkpoint_final.pt
|-- checkpoint_best_observed_reference_l2.pt  # only when a reference exists
`-- run_summary.json
```

The code refuses to overwrite a nonempty run directory. Change `--output-dir`
or the seed before repeating a completed configuration.

Metric schema version 2 records the loss and reference errors from the same model
state after the reported optimizer update. Reference errors use deterministic
evaluation points that are not training nodes:

- Poisson 1D: 2,048 midpoint samples.
- Poisson 2D: an 80 x 80 midpoint grid.
- Heat: 24 equal-area radii x 64 angles x 20 midpoint times.

The reported relative L1 and L2 values are discrete norms on those samples, not
exact continuum integrals. Burgers and Allen-Cahn do not have independent
reference arrays in this repository, so their reference-error fields are empty.

`checkpoint_best_observed_reference_l2.pt` is selected retrospectively from the
logged states. It is a diagnostic checkpoint, not early stopping or an
independently validated model-selection result.

Timing output separates RBF-FD operator construction, optimizer updates,
evaluation, and total process time. CUDA is synchronized at measured optimizer
boundaries.

## Network and collocation configurations

All networks use tanh hidden activations, a scalar linear output, Xavier-normal
weights, and zero biases.

| Problem | Hidden layers | Training cloud |
| --- | --- | --- |
| `poisson1d` | `20, 20, 20` | 129 uniform nodes |
| `poisson2d` | `20, 20, 20` | 21 x 21 Cartesian nodes |
| `burgers` | `32, 32, 32` | 65 spatial nodes x 25 time levels |
| `allen_cahn` | `64, 64, 64, 64` | 65 spatial nodes x 25 time levels |
| `heat` | `50, 50, 50, 50` | 441 physical disk nodes plus 160 ghost nodes x 25 time levels |

The heat ghost nodes support spatial stencils only. PDE and boundary losses are
evaluated on physical nodes as appropriate.

## RBF-FD construction

For a derivative of total order `s` and requested approximation order `p`, the
local polynomial degree is

\[
q = p + s - 1.
\]

In `d` spatial dimensions, the nominal stencil contains

\[
2\binom{q+d}{d}+1
\]

nearest neighbors. Coordinates are centered at the target and scaled by the
farthest stencil distance. The PHS exponent is the smallest odd integer greater
than or equal to `q`, clipped to the interval `[5, 11]`.

Operator construction uses float64 and enforces the following checks:

- Rank-deficient nearest-neighbor stencils are expanded one point at a time.
- Duplicate nodes and non-finite inputs are rejected.
- No pseudoinverse or least-squares fallback is used.
- The default augmented-system condition-number limit is `1e12`.
- Normalized backward error must not exceed `1e-12`.
- Normalized polynomial reproduction error must not exceed `1e-7`.

Operators are converted to the requested training dtype after construction.
Third-derivative discretizations can suffer cancellation in float32; use
`--dtype float64` for numerical-accuracy studies.

## Project layout

```text
.
|-- train.py               # CLI, training, metrics, checkpoints, and plots
|-- methods.py             # Five loss implementations
|-- problems.py            # PDEs, clouds, constraints, and operator routing
|-- networks.py            # Tanh MLP and exact output transformations
|-- rbf_fd.py              # Sparse PHS RBF-FD construction and diagnostics
|-- requirements.txt       # Pinned Python dependencies
`-- tests/
    |-- test_methods.py    # Numerical and end-to-end method checks
    |-- test_reporting.py  # Reporting and checkpoint regression checks
    `-- test1.py           # Long five-method Poisson 1D comparison
```

## Reproducibility notes

- Seeds control Python, NumPy, and PyTorch; deterministic PyTorch algorithms are
  requested with warnings enabled.
- A normal run constructs only the operators required by its selected method.
  gPINN constructs no RBF-FD operators.
- Passing an existing `Problem` object to `run_training` reuses its operator bank
  and reports zero construction time for that run.
- Compare reference errors as well as training losses because the five methods
  optimize different objectives.

## Citation and license

No publication citation or software license is currently declared. Add both
before redistributing the project as a release.
