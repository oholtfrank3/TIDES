# Custom Potentials

An external potential in TiDES is any class with a `calculate_potential` method.
There is no base class to inherit from and no registration step — you define it
in your input file.

## The mechanism

Every potential passed to `add_potential()` is stored in `rt._potential`. During
each Fock build, TiDES calls `calculate_potential(rt_scf)` on each one and adds
the returned array to `fock_ao`:

```python
def apply_potential(self):
    for v_ext in self._potential:
        self.fock_ao += v_ext.calculate_potential(self)
```

Two consequences follow. The returned array must be **in the non-orthogonal AO
basis**, and it must have the **same shape as `fock_ao`** — `(n,n)` for a
restricted reference, `(2,n,n)` for an unrestricted one.

## Template

```python
import numpy as np
from pyscf import gto, scf
from tides import RT_SCF

h2o_mol = gto.M(atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
        ''', basis='6-31G')

h2o = scf.RHF(h2o_mol)
h2o.kernel()

rt_h2o = RT_SCF(h2o, 0.2, 10)

class CUSTOM:
    def __init__(self):
        pass

    def calculate_potential(self, rt_scf):
        # Must return an array in the NON-ORTHOGONAL AO basis
        # with the same shape as rt_scf.fock_ao
        return np.zeros(rt_scf.fock_ao.shape)

rt_h2o.add_potential(CUSTOM())
rt_h2o.kernel()
```

This is `examples/Blank_Potential/Blank_Potential.py`. A filled-in version is
`examples/Custom_Potential/`.

`add_potential` accepts several at once, and they are summed:

```python
rt.add_potential(field, cap, my_potential)
```

## Time dependence

`calculate_potential` receives the RT object, so `rt_scf.current_time` is how a
potential becomes time-dependent. This is exactly how `ElectricField` works:

```python
def resonant_energy(self, rt_scf):
    return self.amplitude * np.sin(self.frequency * rt_scf.current_time + self.phase)
```

A potential that ignores `current_time` is static.

## Non-Hermitian potentials

Returning a complex, non-Hermitian array is legitimate — that is precisely what
a [CAP](../user-guide/potentials.md#complex-absorbing-potential) does, returning
`1j * damping_matrix_ao`.

Be aware of the performance consequence. TiDES decides how to compute the matrix
exponential with:

```python
hermitian = len(rt_scf._potential) == 0
```

Any potential at all — even a strictly Hermitian one like an electric field —
sends the integrators onto the general `scipy.linalg.expm` path rather than the
faster Hermitian eigendecomposition. This is conservative but correct.

## A worked example: MOCAP

The `MOCAP` class in `tides/potentials/rt_cap.py` is a good model. Its structure
generalizes to most non-trivial potentials:

```python
class MOCAP:
    def __init__(self, expconst, emin, prefac=1, maxval=100):
        self.expconst = expconst
        self.emin = emin
        self.prefac = prefac
        self.maxval = maxval

    def calculate_cap(self, rt_scf, fock):
        # 1. Rotate the Fock matrix to the orthogonal basis
        fock_orth = np.dot(rt_scf.orth.T, np.dot(fock, rt_scf.orth))

        # 2. Diagonalize to get MO energies
        mo_energy, mo_orth = np.linalg.eigh(fock_orth)

        # 3. Build the damping terms in that eigenbasis
        # ... (see source)

        # 4. Rotate back to the AO basis before returning
        transform = inv(rt_scf.orth.T)
        damping_matrix_ao = np.dot(transform, np.dot(damping_matrix, transform.T))
        return 1j * damping_matrix_ao

    def calculate_potential(self, rt_scf):
        if rt_scf.nmat == 1:
            return self.calculate_cap(rt_scf, rt_scf.fock_ao)
        else:
            return np.stack((self.calculate_cap(rt_scf, rt_scf.fock_ao[0]),
                             self.calculate_cap(rt_scf, rt_scf.fock_ao[1])))
```

Note the three patterns worth copying:

1. **`calculate_potential` is only a driver.** The real work lives in a helper;
   the driver handles spin structure.
2. **`nmat` dispatch.** Restricted gets one matrix, unrestricted gets a stacked
   pair. Handle both if you want your potential to be general.
3. **Rotate back.** The potential is *built* in whatever basis is convenient,
   but it must be *returned* in the AO basis.

## Static potentials

If a potential is time-independent and your nuclei are fixed, you don't need a
potential class at all — fold it into the core Hamiltonian of the SCF object
before creating the RT object, as
[`static_bfield`](../user-guide/potentials.md#static-magnetic-field) does. For
moving nuclei this doesn't work, because the core Hamiltonian is rebuilt after
each nuclear update; use a potential class instead.
