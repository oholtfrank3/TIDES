# Cite

If you use TiDES in published work, please cite:

> Rohan, M. C.; Suarez, V. A.; Clothier, M.; Kretchmer, J. S.
> **TiDES: A time-dependent electronic structure code for real-time electron and
> spin dynamics.** *The Journal of Chemical Physics* **164**, 234117 (2026).
> [10.1063/5.0336359](https://doi.org/10.1063/5.0336359)

## BibTeX

```bibtex
@article{tides2026,
  title   = {TiDES: A time-dependent electronic structure code for real-time
             electron and spin dynamics},
  author  = {Rohan, Matthew C. and Suarez, Victor A. and Clothier, Mikhayla
             and Kretchmer, Joshua S.},
  journal = {The Journal of Chemical Physics},
  volume  = {164},
  number  = {23},
  pages   = {234117},
  year    = {2026},
  doi     = {10.1063/5.0336359},
}
```

A preprint is available on ChemRxiv under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/):
[10.26434/chemrxiv.15001515](https://doi.org/10.26434/chemrxiv.15001515).

## Please also cite PySCF

TiDES is built on PySCF and depends on it for integrals, SCF, gradients, and
relativistic transformations. If you cite TiDES, cite PySCF as well — see
[PySCF's citation page](https://pyscf.org/about.html).

## Method references

Depending on what you use, these are the primary sources for the underlying
methods:

| Feature | Reference |
|---|---|
| `etrs` integrator | Castro, Marques & Rubio, *J. Chem. Phys.* **121**, 3425 (2004) |
| `ep_pc` integrator | Zhu & Herbert, *J. Chem. Phys.* **148**, 044117 (2018) |
| `cfm4` integrator | Blanes & Moan, *Appl. Numer. Math.* **56**, 1519 (2006); Gómez Pueyo *et al.*, *JCTC* **14**, 3040 (2018) |
| MOCAP | Lopata & Govind, *JCTC* **9**, 4939 (2013) |
| CAP for ionization / ICD | [10.1021/ct400569s](https://doi.org/10.1021/ct400569s) |
| Ehrenfest formulation | Li *et al.*, *J. Chem. Phys.* **123**, 084106 (2005) |
| NOSCF routine | Originally implemented in NWChem |

## Figures

Figures reproduced on this site are taken from the TiDES paper's ChemRxiv
preprint and are used under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## License

TiDES is released under the Apache 2.0 license. See
[LICENSE.txt](https://github.com/oholtfrank3/TIDES/blob/main/LICENSE.txt).
