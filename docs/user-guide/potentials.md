# External Potentials

External potentials are added to the Fock matrix during propagation,

$$\mathbf{F}(t) = \mathbf{F}_0(t) + \mathbf{V}(t)$$

where $\mathbf{F}_0$ is the unperturbed Fock matrix. **All external potentials in
TiDES are added in the AO basis.**

Attach one with `add_potential()`, which accepts any number of potentials:

```python
from tides import RT_SCF, ElectricField

rt = RT_SCF(mf, 0.2, 500)
rt.add_potential(ElectricField('delta', [0.0001, 0.0, 0.0]))
```

Every potential in the list has its `calculate_potential(rt_scf)` method called
at each Fock build, and the returned array is added to `fock_ao`.

## Electric field

```python
ElectricField(field_type, amplitude, center=0, frequency=0, width=0, phase=0)
```

$$\mathbf{V}(t) = -\sum_i \mathbf{D}^i E_i(t)$$

This is the electric dipole approximation: spatial variation of the field is
neglected, which is appropriate whenever the wavelength is much larger than the
molecule.

`amplitude` is the field amplitude along `[x, y, z]` in au. Which of the
remaining parameters matter depends on `field_type`:

| `field_type` | $E(t)$ | Parameters used |
|---|---|---|
| `'delta'` | Impulse applied at a single step | `amplitude`, `center` |
| `'gaussian'` | Gaussian envelope × sine carrier | `amplitude`, `center`, `width`, `frequency`, `phase` |
| `'hann'` | Hann (sin²) window × sine carrier | `amplitude`, `center`, `width`, `frequency`, `phase` |
| `'resonant'` | Continuous sine wave | `amplitude`, `frequency`, `phase` |

The field types follow
[NWChem's excitation rules](https://nwchemgit.github.io/RT-TDDFT.html#excite-excitation-rules).

!!! warning "The delta kick fires on an exact time match"
    `'delta'` applies only when `current_time == center` exactly. With the
    default `center=0` it fires on the first step. If you set a `center` that
    the timestep never lands on exactly, the kick silently never happens.

### Delta kick

```python
ElectricField('delta', [0.0001, 0.0001, 0.0001])
```

An impulse is infinitely narrow in time and therefore infinitely broad in
frequency, so it excites all dipole-allowed transitions at once — the basis of
[real-time spectroscopy](spectroscopy.md). In practice the pulse is one
timestep wide.

Keep the amplitude in the linear-response regime; 0.0001 au is a reasonable
default. Too strong and the response is no longer linear and the resulting
"spectrum" is meaningless.

### Resonant excitation

```python
ElectricField('resonant', [0.0, 0.0, 0.001], frequency=0.3768)
```

A continuous sine wave at a chosen frequency, used to drive a specific
transition. See `examples/Water_ResonantExcitation/`.

### Relativistic references

`ElectricField` detects `DKS`/`DHF` objects and builds the transition dipole in
the 2-spinor basis, including the small-component contribution. No change to
your input is needed.

## Complex absorbing potential

A CAP damps density that reaches the edge of the finite basis, which is how
TiDES treats ionization. TiDES implements the molecular-orbital CAP of Lopata &
Govind:

```python
from tides import MOCAP

cap = MOCAP(expconst=0.5, emin=0.2, prefac=1, maxval=100)
rt.add_potential(cap)
```

$$\mathbf{F}(t) = \mathbf{F}_0(t) + i\Gamma(t)$$

The damping matrix $\Gamma$ is diagonal in the eigenbasis of $\mathbf{F}'$, with
entries non-zero only above a threshold energy $\varepsilon_0$:

$$\gamma_i = \begin{cases} 0, & \varepsilon_i - \varepsilon_0 \le 0 \\ \gamma_0\left(e^{-\xi(\varepsilon_i - \varepsilon_0)} - 1\right), & \varepsilon_i - \varepsilon_0 > 0\end{cases}$$

| Argument | Symbol | Meaning |
|---|---|---|
| `expconst` | $\xi$ | Exponential constant setting how fast damping grows above threshold |
| `emin` | $\varepsilon_0$ | Energy threshold; orbitals below it are undamped |
| `prefac` | $\gamma_0$ | Overall damping strength |
| `maxval` | — | Cap on the magnitude of any damping term |

$\xi$ and $\gamma_0$ are phenomenological and must be chosen for the
application.

!!! note "A CAP makes the Fock matrix non-Hermitian"
    This changes the equation of motion for the density matrix to
    $i\dot{\mathbf{P}}' = \mathbf{F}'\mathbf{P}' - \mathbf{P}'\mathbf{F}'^\dagger$.
    Internally it also forces the integrators onto a general matrix exponential
    instead of the faster Hermitian eigendecomposition.

References: Lopata & Govind, *JCTC* **9**, 4939 (2013); the ionization/ICD
application is
[10.1021/ct400569s](https://doi.org/10.1021/ct400569s).
See `examples/Water_MOCAP/`, which includes a matched run without the CAP.

## Static magnetic field

A static field is time-independent, so it can simply be folded into the core
Hamiltonian:

$$\tilde{\mathbf{h}} = \begin{pmatrix} \mathbf{h}_0 + \tfrac{1}{2}B_z\mathbf{S} & \tfrac{1}{2}(B_x - iB_y)\mathbf{S} \\ \tfrac{1}{2}(B_x + iB_y)\mathbf{S} & \mathbf{h}_0 - \tfrac{1}{2}B_z\mathbf{S}\end{pmatrix}$$

Two interfaces exist:

```python
from tides import static_bfield

# Overwrites the SCF object's hcore, before creating the RT_SCF object
static_bfield(mf, [0, 0, 0.000085])
```

```python
from tides.potentials.rt_vapp import StaticMagneticField

# As a regular external potential
rt.add_potential(StaticMagneticField([0, 0, 0.000085]))
```

Both are for **spin-generalized** (GHF/GKS) references only.

!!! warning "Not available for Ehrenfest"
    `StaticMagneticField` is not currently usable with `RT_Ehrenfest`, pending
    generalized gradient support in PySCF. For moving nuclei a static field
    would need re-adding to the core Hamiltonian after each nuclear update.

See `examples/H_BField/`.

## Custom potentials

Any class with a `calculate_potential(self, rt_scf)` method returning an AO-basis
array can be added — no changes to the TiDES source required. See
[Custom Potentials](../customization/potentials.md).
