# Propagation & Integrators

TiDES propagates the MO coefficients in the orthogonal AO (OAO) basis according
to

$$i\dot{\mathbf{C}}'(t) = \mathbf{F}'(t)\,\mathbf{C}'(t)$$

An equivalent Liouville–von Neumann form propagates the one-electron reduced
density matrix,

$$i\dot{\mathbf{P}}'(t) = [\mathbf{F}'(t), \mathbf{P}'(t)]$$

TiDES propagates $\mathbf{C}'(t)$ rather than $\mathbf{P}'(t)$, since having
explicit access to the MO coefficients at every step is useful for analysis.

## Propagation parameters

```python
rt = RT_SCF(mf, timestep=0.2, max_time=500, frequency=1, verbose=3)
```

| Parameter | Meaning |
|---|---|
| `timestep` | Timestep in atomic units. Constant for the whole run. 1 fs ≈ 41.34 au |
| `max_time` | Total propagation time in atomic units |
| `frequency` | How often observables are printed, in steps. Default `1` (every step) |
| `verbose` | PySCF logger level. Default `3` ("Note"), where most output lives |
| `prop` | Integrator name. Default `'magnus_interpol'` |
| `chkfile` | Restart file. Defaults to `tides.chk` — see [Restarts](restarts.md) |

Select an integrator either at construction or by assignment:

```python
rt = RT_SCF(mf, 0.2, 500)
rt.prop = 'etrs'
```

Self-consistent integrators additionally respect two attributes:

```python
rt.magnus_tolerance = 1e-8   # convergence threshold
rt.magnus_maxiter   = 200    # iteration cap
```

Despite the names, these are read by `etrs`, `ep_pc`, and `cfm4` as well, where
they default to `1e-7` and `20` if unset.

## Integrators for `RT_SCF` and `RT_Ehrenfest`

Six integrators are available. "Fock builds/step" is the dominant cost — a Fock
build is far more expensive than the matrix exponential.

| `prop` | Scheme | Order | Unitary | Self-consistent | Fock builds/step |
|---|---|---|---|---|---|
| `magnus_interpol` | Interpolated Magnus (**default**) | 2 | yes | yes | iterative |
| `magnus_step` | MMUT leapfrog | 2 | yes | no | 1 |
| `rk4` | Runge–Kutta | 4 | no | no | 1 |
| `etrs` | Enforced time-reversal symmetry | 2 | yes | yes | 2+ |
| `ep_pc` | Exponential predictor/corrector | 2 | yes | yes | 2+ |
| `cfm4` | Commutator-free Magnus | 4 | yes | yes | 2+ |

!!! note "Only three of these appear in the TiDES paper"
    The paper describes `rk4`, `magnus_step` (MMUT), and `magnus_interpol`.
    `etrs`, `ep_pc`, and `cfm4` were added afterwards and are documented here
    because they are present and callable in this version of the code.

### `magnus_interpol` — interpolated Magnus (default)

Propagates with the Fock matrix at the midpoint of the step,
$\mathbf{C}'(t+\Delta t) = \mathbf{U}(t + \tfrac{\Delta t}{2})\,\mathbf{C}'(t)$
with $\mathbf{U} = \exp[-i\,\Delta t\,\mathbf{F}'(t+\tfrac{\Delta t}{2})]$.

The midpoint Fock is not known in advance, so it is found self-consistently:

1. Extrapolate $\mathbf{F}'(t + \tfrac{\Delta t}{2}) = 2\mathbf{F}'(t) - \mathbf{F}'(t - \tfrac{\Delta t}{2})$
2. Propagate to get $\mathbf{C}'(t+\Delta t)$
3. Build $\mathbf{F}'(t+\Delta t)$ from the new coefficients
4. Interpolate $\mathbf{F}'(t+\tfrac{\Delta t}{2}) = \tfrac{1}{2}\mathbf{F}'(t) + \tfrac{1}{2}\mathbf{F}'(t+\Delta t)$
5. Repeat 2–4 until converged

Building several Fock matrices per step buys accuracy at large timesteps —
this integrator maintains accuracy at Δt ≈ 0.5–1.0 au where the explicit
schemes require ≈ 0.05 au. If it fails to converge it logs an error telling you
to raise `magnus_maxiter` or lower `timestep`.

### `magnus_step` — MMUT

A leapfrog step,
$\mathbf{C}'(t+\Delta t) = \exp[-i\,2\Delta t\,\mathbf{F}'(t)]\,\mathbf{C}'(t-\Delta t)$,
using the current Fock to jump across $2\Delta t$. One Fock build per step and
no self-consistency, so it is cheap per step but not self-consistent, and
energy drifts. It requires small timesteps (≈ 0.05 au). This is the integrator
used in the original magnetization-dynamics work reproduced in
`examples/H_BField/`.

### `rk4` — Runge–Kutta

Standard RK4 on the MO coefficients. Note that it uses the Fock matrix from the
start of the step for all four stages — there is no midpoint Fock update.

RK4 is **not unitary**: the MO coefficients lose orthonormality as the
propagation proceeds. TiDES applies a QR re-orthogonalization after every step
to prevent the norm from blowing up, but this does not restore energy
conservation. Use small timesteps, and prefer a unitary integrator unless you
have a specific reason not to.

### `etrs` — enforced time-reversal symmetry

$$\mathbf{C}'(t+\Delta t) = e^{-i\frac{\Delta t}{2}\mathbf{F}'(t+\Delta t)}\,e^{-i\frac{\Delta t}{2}\mathbf{F}'(t)}\,\mathbf{C}'(t)$$

$\mathbf{F}'(t+\Delta t)$ depends on $\mathbf{C}'(t+\Delta t)$, so the step is
iterated to self-consistency from a linearly extrapolated predictor. Unitary and
time-reversible by construction.

Reference: Castro, Marques & Rubio, *J. Chem. Phys.* **121**, 3425 (2004).

### `ep_pc` — exponential predictor/corrector

An EP-PC1 predictor/corrector. Each step takes a full MMUT step as a predictor
(requiring no Fock build), builds the Fock matrix from the predicted density,
then corrects using a propagator built from the trapezoidal average
$\tfrac{1}{2}(\mathbf{F}_N + \mathbf{F}^p)$, iterating until the predicted and
corrected densities agree.

Reference: Zhu & Herbert, *J. Chem. Phys.* **148**, 044117 (2018), Algorithm 2.

### `cfm4` — commutator-free Magnus, 4th order

$$\phi(t+\Delta t) = e^{-i\Delta t(\alpha_1 \mathbf{F}_1 + \alpha_2 \mathbf{F}_2)}\,e^{-i\Delta t(\alpha_2 \mathbf{F}_1 + \alpha_1 \mathbf{F}_2)}\,\phi(t)$$

$\mathbf{F}_1$ and $\mathbf{F}_2$ are the Fock matrices at the two
Gauss–Legendre quadrature nodes $c_{1,2} = \tfrac{1}{2} \mp \tfrac{\sqrt{3}}{6}$,
obtained by linear interpolation between $\mathbf{F}(t)$ and
$\mathbf{F}(t+\Delta t)$, with $\mathbf{F}(t+\Delta t)$ determined
self-consistently. Fourth-order accurate once the self-consistency converges.

References: Blanes & Moan, *Appl. Numer. Math.* **56**, 1519 (2006);
Gómez Pueyo *et al.*, *JCTC* **14**, 3040 (2018).

## Integrators for `RT_CAS_RAS`

!!! danger "Separate, experimental, and not interchangeable"
    `RT_CAS_RAS` has its own propagation loop and its own integrators. They are
    **not** interchangeable with the six above. All eight names live in the same
    `INTEGRATORS` dictionary, but the two families have different call
    signatures and there is no guard against mixing them: setting
    `prop='rk4cr'` on an `RT_SCF` object, or `prop='cfm4'` on an
    `RT_CAS_RAS` object, raises a bare `TypeError` about positional arguments.

| `prop` | Scheme | Status |
|---|---|---|
| `rk4cr` | 4th-order Runge–Kutta for CAS/RAS | Experimental |
| `vv` | Velocity Verlet, CASCI only | **Currently broken** — see below |

!!! bug "`vv` does not run"
    `vv` accepts five arguments, but the CAS propagation loop
    (`tides/propagators/rt_casprop.py`) calls the integrator with six. Setting
    `prop='vv'` fails immediately with
    `vv() takes 5 positional arguments but 6 were given`.
    Use `rk4cr` until this is fixed.

`vv` is additionally documented in the source as CASCI-only — it is not
implemented for CASSCF.

See [CAS/RAS Dynamics](cas-ras.md) for how to set these calculations up.

## Choosing a timestep

The timestep is fixed for the whole run and interacts strongly with the choice
of integrator:

- Explicit schemes (`magnus_step`, `rk4`) need small timesteps — around
  0.05 au — because they use a single Fock matrix per step.
- Self-consistent schemes (`magnus_interpol`, `etrs`, `ep_pc`, `cfm4`) build
  several Fock matrices per step and remain accurate at Δt ≈ 0.5–1.0 au.

There is a cost trade-off in both directions: a self-consistent integrator costs
more per step but permits far larger steps. The right balance depends on the
system and the observable, and is worth testing for a new class of problem.

For a spectrum, the timestep also sets the maximum resolvable frequency,
$\omega_{\max} = \pi/\Delta t$, and the total time sets the resolution,
$\Delta\omega = 2\pi/t_{\max}$. Core-level (XAS) simulations need small
timesteps to resolve high-frequency transitions — see
[Spectroscopy](spectroscopy.md).
