# Relativistic Methods

Because TiDES propagates whatever Fock matrix the PySCF object produces, most
relativistic treatments available in PySCF work with no change to your TiDES
input. Decorate the SCF object, converge it, and hand it over.

TiDES works with 1-component (spin-restricted) and 2-component
(spin-generalized) frameworks, plus full 4-component references.

## X2C1e

The exact two-component method decouples the Dirac equation into a
two-component picture. PySCF uses the Foldy–Wouthuysen X2C1e transformation,
which transforms only the one-electron integrals and yields a time-independent
transformation matrix. The transformation is applied once at the start to
produce dressed one-electron integrals; the real-time equations of motion are
then exactly as in the non-relativistic case.

```python
from pyscf import gto, scf
from tides import RT_SCF, ElectricField

mol = gto.M(atom='Hg 0 0 0', basis='sapporo-dkh3-dzp-2012-diffuse', spin=0)

mf = scf.GHF(mol).x2c1e()      # decorate, then converge
mf.kernel()

rt = RT_SCF(mf, timestep=0.1, max_time=10000)
rt.observables.update(dipole=True)
rt.add_potential(ElectricField('delta', [0.0001, 0.0, 0.0]))
rt.kernel()
```

This is how spin–orbit coupling enters, and it is enough to capture
normally spin-forbidden singlet–triplet transitions — see
[Spectroscopy](spectroscopy.md#spin-forbidden-transitions-via-soc).

!!! warning "Picture change affects external potentials"
    The X2C1e transformation is a "picture change": operators must be rotated
    accordingly. This matters when defining external potentials, whose integrals
    need to account for it. TiDES handles this for `ElectricField` with
    `DKS`/`DHF` references, but a hand-written
    [custom potential](../customization/potentials.md) must account for it
    itself.

Two-electron SOC terms are not included — PySCF does not implement the X2C2e
transformation. The paper notes this as the main source of difference from
earlier work that added an empirical two-electron correction.

Examples: `examples/Water_GKSX2C1e_UV-Vis/` and
`examples/ExamplesFromOriginalTiDESPaper/X2C1e/`.

## Four-component

Full 4-component Dirac–Hartree–Fock and Dirac–Kohn–Sham references work
directly:

```python
from pyscf import gto, scf

mf = scf.DHF(mol)
mf.kernel()

rt = RT_SCF(mf, 0.1, 1000)
```

`ElectricField` detects `DHF`/`DKS` objects and builds the transition dipole in
the 2-spinor basis, adding the small-component contribution scaled by
$1/(2c)^2$.

Examples: `src/examples/4C/` (`Water_DHF_UV-Vis.py`, `Water_DKS_UV-Vis.py`) and
`src/examples/pNA_polarizability/`.

## ZORA

The zeroth-order regular approximation is implemented natively in TiDES rather
than inherited from PySCF. It works by overwriting the core Hamiltonian:

```python
import pyscf
from pyscf import scf
from tides.zora.relativistic import ZORA

mol = pyscf.gto.M(...)
mf = scf.ghf.GHF(mol)

zora_obj = ZORA(mol)
Hcore = zora_obj.get_zora_correction()
mf.get_hcore = lambda *args: Hcore

mf.kernel()
```

!!! danger "Generalized references only"
    ZORA in TiDES must only be used with spin-generalized (GHF/GKS)
    references.

Because ZORA replaces `get_hcore`, it composes poorly with anything else that
overwrites the core Hamiltonian — notably
[`static_bfield`](potentials.md#static-magnetic-field). Apply the field through
the `StaticMagneticField` potential class instead if you need both.

Examples: `src/examples/zora/` (`Ti-GHF.py`, `Ti-GKS.py`, `VF3-GHF.py`).

## Which to use

| Method | Cost | Notes |
|---|---|---|
| X2C1e | Low | 1-electron SOC; picture-change caveats; well tested with TiDES |
| 4-component | High | Reference-quality; no picture-change ambiguity |
| ZORA | Low | Native to TiDES; generalized references only |
