# Observables

Nothing is computed unless you ask for it. Every observable is a key in the
`observables` dictionary on the `RT_SCF` object, and all default to `False`.

```python
rt.observables['energy'] = True

# or several at once
rt.observables.update(energy=True, dipole=True, mulliken_atom_charge=True)
```

Enabled observables are computed and printed every `frequency` steps. Most print
at the default verbosity (`verbose=3`, "Note"); the exceptions are noted below.

## Available observables

| Key | Prints |
|---|---|
| `energy` | Total energy in au. For Ehrenfest, also total kinetic energy; per-atom kinetic energies at `verbose>3` |
| `dipole` | Total dipole moment `[X, Y, Z]` in au |
| `quadrupole` | Total quadrupole moment, full 3×3 tensor, in au |
| `charge` | Total electronic charge |
| `mulliken_charge` | Mulliken atomic charges |
| `mulliken_atom_charge` | Mulliken atomic charges (same function as above) |
| `hirsh_charge` | Hirshfeld atomic charges |
| `hirsh_atom_charge` | Hirshfeld atomic charges (same function as above) |
| `hirshi_atom_charge` | Hirshfeld-I charges, via HORTON |
| `plane_partition_charge` | Charge partitioned across a plane |
| `plane_partition_charge_spatial` | Plane-partitioned charge by spatial integration |
| `mag` | Total magnetization `[X, Y, Z]` |
| `hirsh_mag` | Hirshfeld atomic magnetization |
| `hirsh_atom_mag` | Hirshfeld atomic magnetization (same function as above) |
| `spin_square` | ⟨S²⟩ and the multiplicity 2S+1 |
| `mo_occ` | MO occupations |
| `mo_occ_separate` | MO occupations resolved into alpha and beta |
| `nuclei` | Nuclear positions → `trajectory.xyz`. Ehrenfest only |
| `cube_density` | Density written to a cube file |
| `mo_coeff` | Raw MO coefficients |
| `den_ao` | Density matrix in the AO basis |
| `fock_ao` | Fock matrix in the AO basis |

!!! note "Duplicate keys are intentional"
    `mulliken_charge`/`mulliken_atom_charge`, `hirsh_charge`/`hirsh_atom_charge`,
    and `hirsh_mag`/`hirsh_atom_mag` each map to the same underlying function.
    Either spelling works.

## Dipole moment

The total dipole in Cartesian direction $i$ is

$$\mu_i(t) = \sum_{jk}\left(\mathbf{P}^{\alpha\alpha}_{jk}(t) + \mathbf{P}^{\beta\beta}_{jk}(t)\right)\mathbf{D}^i_{kj} + \sum_A Q_A R^i_A$$

where $\mathbf{D}^i$ is the electronic transition dipole matrix and the second
term is the nuclear contribution. This is the observable you need for
[spectroscopy](spectroscopy.md).

For fixed nuclei $\mathbf{D}$ is time-independent; for Ehrenfest it is
recomputed after every nuclear update.

## Magnetization

The spin magnetization density vector is

$$\vec{\mathbf{M}}(\mathbf{r}) = \mathrm{Tr}[\vec{\sigma}\rho(\mathbf{r})]$$

with $\vec\sigma$ the vector of Pauli matrices. Integrating over all space gives
the total magnetization:

$$M_x = \sum_{ij}[\mathbf{P}^{\alpha\beta}_{ij}(t) + \mathbf{P}^{\beta\alpha}_{ij}(t)]\mathbf{S}_{ij}$$

$$M_y = \sum_{ij} i[\mathbf{P}^{\alpha\beta}_{ij}(t) - \mathbf{P}^{\beta\alpha}_{ij}(t)]\mathbf{S}_{ij}$$

$$M_z = \sum_{ij}[\mathbf{P}^{\alpha\alpha}_{ij}(t) - \mathbf{P}^{\beta\beta}_{ij}(t)]\mathbf{S}_{ij}$$

$M_x$ and $M_y$ depend on the off-diagonal spin blocks, which are non-zero only
in a **spin-generalized** (GHF/GKS) calculation. $M_z$ is available for
unrestricted references too.

Requesting `mag` or `hirsh_atom_mag` on a non-generalized reference raises an
error at the start of propagation.

See [`examples/H_BField/`](../examples.md) and the Li-trimer spin-precession
example for worked cases.

## Hirshfeld partitioning

The Hirshfeld observables require an external module:

```bash
pip install git+https://github.com/frobnitzem/hirshfeld
```

If it isn't installed, TiDES prints a note at import time and the Hirshfeld
observables are unavailable. `hirshi_atom_charge` (Hirshfeld-I) additionally
requires HORTON.

## MO occupations

MO occupations come from the diagonal of

$$\mathbf{P}_{\mathrm{MO}} = \tilde{\mathbf{C}}^\dagger \mathbf{S}\mathbf{P}(t)\tilde{\mathbf{C}}$$

where $\tilde{\mathbf{C}}$ is any orthonormal MO basis. By default this is the
ground-state SCF basis, $\tilde{\mathbf{C}} = \mathbf{C}_{SCF}$. Projecting onto
a *fragment* basis instead is often far more informative for charge-transfer and
decay processes — see [Fragments & NOSCF](fragments.md).

## Nuclei (Ehrenfest)

Setting `nuclei=True` on an `RT_Ehrenfest` object writes an XYZ trajectory to
`trajectory.xyz` rather than to the main output.

| Quantity | Verbosity required |
|---|---|
| Positions | `verbose>2` (default) |
| Velocities | `verbose>3` |
| Forces | `verbose>4` |

## Reading the output

`tides.analysis.parse_rt.parse_output` parses an output file into a dictionary
of NumPy arrays:

```python
from tides.analysis.parse_rt import parse_output

result = parse_output('Water_RHF_UV-Vis.out')
result['time']      # (nsteps,)
result['dipole']    # (nsteps, 3)
```

Available keys include `time`, `energy`, `kinetic_energy`, `dipole`,
`quadrupole`, `charge`, `atom_charge`, `mulliken_charge`,
`mulliken_atom_charge`, `hirsh_charge`, `hirsh_atom_charge`, `mag`,
`hirsh_mag`, `hirsh_atom_mag`, `mo_occ`, `mo_occ_alpha`, `mo_occ_beta`,
`frag_charge`, `plane_partition_charge`, `plane_partition_charge_spatial`,
`alpha_energies`, `beta_energies`, and `spin_square`.

!!! warning "Charges are electron counts, not formal charges"
    The charge observables report the **electronic** charge on each atom. To get
    a formal charge, subtract from the nuclear charge — e.g. for lithium,
    `3 - result['mulliken_atom_charge'][:,0]`. Every `Workup_*.py` script in the
    examples does this.

## Adding your own

Any quantity you can compute from the `RT_SCF` object can be added as an
observable without modifying TiDES — see
[Custom Observables](../customization/observables.md).
