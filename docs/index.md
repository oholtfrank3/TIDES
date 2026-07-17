![TiDES](assets/logo.png){ width="450" }

**Ti**me-**D**ependent **E**lectronic **S**tructure — an open-source package for
real-time electronic structure simulations, built on top of
[PySCF](https://pyscf.org).

TiDES propagates the electronic density matrix in real time, giving direct access
to observables that are awkward or impossible to obtain from linear-response
methods: charge migration, non-perturbative field response, and coupled
electron–nuclear dynamics via Ehrenfest.

## Capabilities

| Area | Notes |
|---|---|
| RT-TDHF / RT-TDDFT | Restricted, unrestricted, and generalized references (RHF/UHF/GHF, RKS/UKS/GKS) |
| Ehrenfest dynamics | `RT_Ehrenfest` — coupled electron–nuclear propagation |
| Relativistic | X2C1e via PySCF |
| Complex absorbing potentials | Molecular-orbital CAP for autoionizing / metastable states |
| Spectroscopy | UV-Vis, XAS from dipole autocorrelation |
| Observables | Mulliken and Hirshfeld charges, magnetization, energy, dipole |
| Extensibility | Custom observables and external potentials |

## The basic workflow

1. Build a PySCF `mol`
2. Create and run an SCF object (RHF/UHF/GHF/RKS/UKS/GKS)
3. Wrap it in `RT_SCF` or `RT_Ehrenfest` with propagation parameters
4. Declare observables
5. Add any external potentials
6. Call `.kernel()`

See [Installation](getting-started/installation.md) to get set up, or
[First Calculation](getting-started/first-calculation.md) for a worked example.

## Examples

The [`examples/`](https://github.com/jskretchmer/TIDES/tree/main/examples)
directory in the repository contains runnable scripts for every major feature,
including all figures from the original TiDES paper. `examples/Example_Workbook.ipynb`
is a good interactive starting point.
