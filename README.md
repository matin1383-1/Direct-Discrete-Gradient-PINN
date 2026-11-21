# Direct Discrete Training of Gradient-Enhanced Physics-Informed Neural Networks

Standalone PyTorch code accompanying the AI-SOFT 2026 paper by Mahdi Zekavat
Bushehri, Amirhossein Ayoubi, and Mohsen Yousefnezhad.

The paper proposes **Direct Discrete-Gradient Physics-Informed Neural Networks
(DDG-PINNs)**. DDG-PINN analytically differentiates the continuous PDE residual,
then independently discretizes the spatial differential terms in the residual
and its gradient with sparse polyharmonic-spline radial-basis-function finite
differences (PHS RBF-FD). It does not apply a first-derivative matrix to an
already discretized residual.

## Paper scope and repository scope

The main paper evaluates one benchmark: the one-dimensional Poisson equation.
The paper-reproduction path in this repository follows that protocol.

| Scope | Problems | Methods |
| --- | --- | --- |
| Main paper | `poisson1d` | PINN, gPINN, DT-PINN orders 3/4, DDG-PINN order 3 |
| Exploratory extensions, not paper results | `poisson2d`, `burgers`, `allen_cahn`, `heat` | The same compatible formulations plus DT-PINN order 5 |

Results from the extension problems must not be presented as results from the
paper. The paper's gPINN timing baseline used the authors' DeepXDE
implementation; this standalone repository implements every method in PyTorch.
Consequently, numerical trends are comparable, but exact runtimes and optimizer
trajectories can differ by framework, hardware, and software version.

## Mathematical formulation

The paper considers

```math
u''(x)=b(x), \qquad x\in(0,\pi),
```

with $u(0)=0$ and $u(\pi)=\pi$. The boundary conditions are enforced exactly by

```math
u_\theta(x)=x+\tanh(x)\tanh(\pi-x)N_\theta(x).
```

For $K=\{1,2,3,4,8\}$, the manufactured solution and forcing are

```math
u(x)=x+\sum_{k\in K}\frac{\sin(kx)}{k},
\qquad
b(x)=-\sum_{k\in K}k\sin(kx),
\qquad
b'(x)=-\sum_{k\in K}k^2\cos(kx).
```

The continuous residual and its derivative are

```math
r_\theta(x)=u_\theta''(x)-b(x),
\qquad
r_\theta'(x)=u_\theta'''(x)-b'(x).
```

At $N$ collocation points, PINN and gPINN minimize

```math
\begin{aligned}
\mathcal{L}_{\mathrm{PINN}}(\theta)
  &=\frac{1}{N}\sum_{i=1}^{N}r_\theta(x_i)^2,\\
\mathcal{L}_{\mathrm{gPINN}}(\theta)
  &=\frac{1}{N}\sum_{i=1}^{N}
    \left[r_\theta(x_i)^2+\lambda r_\theta'(x_i)^2\right].
\end{aligned}
```

Let $D_2$ and $D_3$ be independently assembled RBF-FD matrices and let
$\boldsymbol{u}_\theta$ contain the network values on the support nodes. DDG-PINN
uses

```math
\boldsymbol{r}_h(\theta)=D_2\boldsymbol{u}_\theta-\boldsymbol{b}_X,
\qquad
\boldsymbol{g}_h(\theta)=D_3\boldsymbol{u}_\theta-\boldsymbol{b}'_X,
```

and minimizes

```math
\mathcal{L}_{\mathrm{DDG}}(\theta)
=\frac{1}{N}\lVert\boldsymbol{r}_h(\theta)\rVert_2^2
+\frac{\omega}{N}\lVert\boldsymbol{g}_h(\theta)\rVert_2^2.
```

The CLI option `--lambda-weight` represents $\lambda$ for gPINN and $\omega$
for DDG-PINN.

## Paper experiment protocol

`paper_experiment.py` encodes Section IV of the paper:

| Item | Paper setting |
| --- | --- |
| Interior collocation points | $x_i=i\pi/16$, $i=1,\ldots,15$ |
| Discrete support nodes | 65 uniform nodes on $[0,\pi]$, including endpoints |
| DDG stencils | 11 nodes for $D_2$ and 13 nodes for $D_3$ |
| Network | Three hidden layers of 20 tanh neurons |
| Initialization | Glorot uniform weights |
| Arithmetic | Float32 |
| Optimizer | Full-batch Adam, constant learning rate $10^{-3}$ |
| Training | 20,000 updates; seeds 200, 201, and 202 |
| gPINN weight | $\lambda=0.1$ |
| DDG-PINN weights | $\omega\in\{0.01,0.1,1\}$ |
| Validation | 1,000 uniform random points, seed 321; every 1,000 updates |
| Final test | Separate uniform grid of 1,001 points |
| Reported aggregate | Arithmetic mean and sample standard deviation over three seeds |

The seven final configurations are PINN; gPINN with $\lambda=0.1$; DDG-PINN
with $\omega=0.01$, $1$, and $0.1$; and DT-PINN orders 3 and 4.

## Installation

The pinned dependency stack is tested in CI with Python 3.11.

```bash
git clone <repository-url>
cd Direct-Discrete-Gradient-PINN
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the verification suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Reproduce the paper comparison

Check the complete seven-configuration pipeline with one update and one seed:

```bash
python paper_experiment.py --smoke-test --output-dir outputs/paper_smoke
```

Run the paper protocol:

```bash
python paper_experiment.py --device cuda
```

The default run performs 21 trainings: seven configurations times three seeds.
Use the same NVIDIA GeForce RTX 3050 and software environment as the paper if
you need a meaningful runtime comparison.

The driver writes:

```text
outputs/paper_reproduction/
|-- protocol.json
|-- individual_results.json
|-- paper_table.csv
|-- paper_table.json
|-- validation_relative_l2.png
`-- <configuration>/poisson1d_<method>_seed<seed>/...
```

`paper_table.csv` contains the final test relative $L^2$ mean and sample standard
deviation and the optimizer-update time mean and sample standard deviation.
Operator construction time is recorded separately.

## Run one configuration

The paper methods use these CLI names:

| CLI method | Formulation |
| --- | --- |
| `pinn` | AD residual only |
| `gpinn` | AD residual and AD residual gradient |
| `dt3` | Order-3 RBF-FD residual |
| `dt4` | Order-4 RBF-FD residual |
| `ddg` | Order-3 RBF-FD residual and independently discretized analytical gradient |

Examples:

```bash
python train.py --method pinn --problem poisson1d --smoke-test
python train.py --method gpinn --problem poisson1d --lambda-weight 0.1
python train.py --method ddg --problem poisson1d --lambda-weight 0.1
python train.py --method dt4 --problem poisson1d
```

Run `python train.py --help` for all options. Normal runs refuse to overwrite a
nonempty run directory.

## RBF-FD construction

For derivative order $q$ and requested approximation order $p$, the polynomial
degree is

```math
\ell_q=p+q-1.
```

In one dimension, the nominal stencil size is

```math
n_q=2(\ell_q+1)+1.
```

Thus the paper's order-3 DDG formulation uses $\ell_2=4$, $\ell_3=5$,
$n_2=11$, and $n_3=13$. Coordinates are centered at the target and scaled by
the farthest stencil distance. The PHS exponent is the smallest odd integer at
least as large as the polynomial degree, clipped to $[5,11]$; both paper DDG
operators therefore use $m=5$.

Operator construction uses float64 and rejects duplicate nodes, non-finite
inputs, unresolved rank deficiencies, inaccurate solves, and failed polynomial
reproduction. Operators are converted to the training dtype after construction.

## Exploratory extensions

The repository also includes Poisson 2D, Burgers, Allen-Cahn, a manufactured
disk heat problem, and DT-PINN order 5. These are implementation extensions
motivated by the paper's future-work section; they were not evaluated in the
paper. For example:

```bash
python train.py --method ddg --problem poisson2d
python train.py --method dt5 --problem heat --dtype float64
```

## Project layout

```text
.
|-- paper_experiment.py    # Seven-configuration paper protocol
|-- train.py               # Single-run CLI, metrics, checkpoints, and plots
|-- methods.py             # PINN, gPINN, DT-PINN, and DDG-PINN losses
|-- problems.py            # Paper benchmark and exploratory PDE extensions
|-- networks.py            # Tanh MLP and exact output transformations
|-- rbf_fd.py              # Sparse PHS RBF-FD construction and diagnostics
|-- CITATION.cff           # Paper citation metadata
|-- requirements.txt       # Pinned runtime dependencies
`-- tests/                 # Numerical, execution, and reporting checks
```

## Citation and license

Citation metadata for the accompanying AI-SOFT 2026 paper is provided in
`CITATION.cff`. A software license has not yet been declared; add one before
redistributing the repository as a software release.
