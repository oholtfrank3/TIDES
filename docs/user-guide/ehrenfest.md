# Ehrenfest Dynamics

`RT_Ehrenfest` couples electronic and nuclear motion beyond the
Born–Oppenheimer approximation. Nuclei move classically on the mean field of the
*non-stationary* electronic state, while the electrons are propagated with the
same RT-TDHF/RT-TDDFT equations of motion used by `RT_SCF`.

`RT_Ehrenfest` derives from `RT_SCF`, so everything in
[Integrators](integrators.md) and [Observables](observables.md) applies.

```python
from pyscf import gto, scf
from tides import RT_Ehrenfest

mol = gto.M(atom='H 0 0 0; H 0 0 0.75', basis='6-31G')
uhf = scf.UHF(mol)
uhf.kernel()

rt = RT_Ehrenfest(uhf, timestep=0.05, max_time=500, Ne_step=1, N_step=1)
rt.observables.update(energy=True, nuclei=True)
rt.kernel()
```

## Mixed timestepping

Nuclear motion is orders of magnitude slower than electronic motion, so the
nuclear degrees of freedom do not need updating every electronic step. Two
parameters control this:

| Parameter | Default | Meaning |
|---|---|---|
| `Ne_step` | `10` | Nuclear positions/velocities are updated every `Ne_step` **electronic** steps |
| `N_step` | `10` | Nuclear forces are updated every `N_step` **nuclear** steps |

So with the defaults, forces are rebuilt every 100 electronic steps. Setting
`Ne_step=1, N_step=1` updates coordinates, velocities, and forces at every
electronic step — the most accurate and most expensive choice.

!!! note "Naming"
    The TiDES paper's Fig. 3 labels these `Nn_step` and `Nf_step`. The code uses
    `Ne_step` and `N_step`; those are the names to use in your input.

The `examples/ExamplesFromOriginalTiDESPaper/Ehrenfest/` directory sweeps these
(`Nn1Nf1`, `Nn2Nf3`, `Nn3Nf2`, …) so you can see the effect directly.

## Forces

The force is the negative gradient of the non-stationary HF/DFT electronic
energy,

$$\frac{\partial E}{\partial R^i_A} = \frac{\partial V_{NN}}{\partial R^i_A} + \mathrm{Tr}\left[\frac{\partial \mathbf{h}}{\partial R^i_A}\mathbf{P} + \frac{1}{2}\frac{\partial \mathbf{V}_{\mathrm{eff}}}{\partial R^i_A}\bigg|_P \mathbf{P}\right] - \mathrm{Tr}\left[\mathbf{F}\mathbf{X}\frac{\partial \mathbf{X}^{-1}}{\partial R^i_A}\mathbf{P} + \mathbf{P}\frac{\partial \mathbf{X}^{-1}}{\partial R^i_A}\mathbf{X}\mathbf{F}\right]$$

This differs from a Born–Oppenheimer gradient because the electronic state is
not stationary. TiDES uses PySCF's native gradient modules for everything except
the orthogonalization-matrix derivative.

!!! note "Löwdin orthogonalization is used for Ehrenfest"
    `RT_SCF` defaults to canonical orthogonalization, but `RT_Ehrenfest` uses
    Löwdin symmetric orthogonalization, $\mathbf{X} = \mathbf{S}^{-1/2}$, because
    it is numerically more stable for nuclear dynamics. This is why
    `RT_Ehrenfest` ignores the `orth` argument.

Ehrenfest is available for spin-restricted, unrestricted, and generalized
references. Note that PySCF has no `grad.GKS`, so TiDES uses `grad.UKS` for
generalized Kohn–Sham gradients.

## Setting initial conditions

Nuclear velocities are set directly on the `nuc` object before propagation:

```python
import numpy as np

# Give the H-H bond 10 eV of vibrational energy
KE_i = 5.0                                    # eV per atom
init_velo = np.sqrt(2 * (KE_i / 27.2114) / 1836)

rt.nuc.vel[0, 2] = -init_velo                 # opposite directions
rt.nuc.vel[1, 2] = +init_velo
```

`rt.nuc.vel` is `(natom, 3)` in atomic units. Positions are `rt.nuc.pos`.

## Trajectory output

With `nuclei=True`, nuclear coordinates are written to `trajectory.xyz` rather
than to the main output file. Velocities are included at `verbose>3` and forces
at `verbose>4`.

Read it back with any XYZ reader:

```python
from MDAnalysis.coordinates.XYZ import XYZReader

xyz = XYZReader('trajectory.xyz', dt=0.05)
xyz.units['time'] = 'au'
```

`tides.analysis.parse_rt.get_length` computes bond distances from the resulting
position array.

## Energy conservation

Total energy — electronic plus nuclear kinetic — should be conserved. It is the
most useful diagnostic you have: if it drifts, your timestep is too large, your
integrator is not self-consistent, or your `Ne_step`/`N_step` are too coarse.
Enable `energy=True` and plot it.

## Worked example: Cl₂ dissociation

![Cl2 Ehrenfest vs AIMD](../assets/paper/fig11_cl2_ehrenfest.png){ width="420" }
/// caption
Cl–Cl bond distance at various initial kinetic energies (0 eV blue; 1 eV orange;
2 eV green; 3 eV red) in (a) 6-31G and (b) 6-31G*. Solid lines are non-adiabatic
*ab initio* Ehrenfest trajectories; dashed lines are adiabatic AIMD on the
ground-state singlet surface. Figure from Rohan *et al.* (2026),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
///

Ehrenfest and AIMD agree at 0 eV and diverge as the initial kinetic energy
grows, since higher energies sample more non-adiabatic character. For both
bases, Cl₂ is more weakly bound in the Ehrenfest simulations.

Inputs: `examples/ExamplesFromOriginalTiDESPaper/Ehrenfest/`. Simpler starting
points are `examples/H2_Ehrenfest/` and `examples/NaCl_Ehrenfest/`, and
`examples/H2_2c-Ehrenfest/` compares BOMD against Ehrenfest for restricted,
unrestricted, and 2-component references.

## Limitations

- Ehrenfest is a mean-field method. When a system should branch between distinct
  surfaces, Ehrenfest instead follows an unphysical average of them.
- The traveling-basis terms arising from atom-centered functions on moving
  nuclei are neglected, consistent with previous work, on the grounds that they
  make a negligible numerical difference.
- [`StaticMagneticField`](potentials.md#static-magnetic-field) is not currently
  available for Ehrenfest.
