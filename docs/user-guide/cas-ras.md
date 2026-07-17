# CAS/RAS Dynamics

`RT_CAS_RAS` performs TD-CASCI and TD-CASSCF calculations, propagating CI
coefficients rather than a single-determinant density matrix. It projects out
the virtual space.

!!! warning "Experimental"
    `RT_CAS_RAS` is newer and less exercised than `RT_SCF`. It has its own
    propagation loop, its own integrators, and several constructor arguments
    that exist only for signature compatibility with `RT_SCF` and are unused.
    TD-RAS-SCF is **not** implemented.

## Usage

```python
from pyscf import gto, scf, mcscf
from tides import RT_CAS_RAS

mol = gto.M(atom=..., basis='6-31G')
mf = scf.RHF(mol)
mf.kernel()

# Solve the CASSCF/CASCI object for its t=0 state first
mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.kernel()

rt = RT_CAS_RAS('CASSCF', mc, timestep, max_time,
                'output.txt', 'corrdens.txt',
                mo_to_ao=..., orth=..., ovlp=mol.intor('int1e_ovlp'))
rt.prop = 'rk4cr'
rt.kernel()
```

## Arguments

| Argument | Meaning |
|---|---|
| `opt` | `'CASCI'` for TD-CAS/RAS-CI, `'CASSCF'` for TD-CAS/RAS-SCF |
| `ras` | A PySCF `mcscf` CASSCF/CASCI object, already solved for its t=0 state |
| `timestep`, `max_time` | As for `RT_SCF` |
| `outputName` | Output file used to check numerical stability |
| `corrDenName` | Output file recording each AO occupation at each timestep |
| `reg` | Minimum allowed eigenvalue before inversion. Only affects CAS/RAS-SCF |
| `opts` | Allowed excitations (1 allows -S, 2 allows -D, …). Leave as default for full CAS |
| `mo_to_ao` | MO→AO transformation. Columns sorted `(core, active)` — **do not include virtuals** |
| `orth` | Orthogonal AO coefficient matrix |
| `ovlp` | AO overlap matrix |
| `h1e`, `h2e` | One- and two-electron Hamiltonians in the AO basis at t=0 |

`filename`, `chkfile`, and `verbose` are accepted but unused; they exist to keep
the signature consistent with `RT_SCF`.

## Integrators

`RT_CAS_RAS` uses a **separate** set of integrators from `RT_SCF`. See
[Integrators](integrators.md#integrators-for-rt_cas_ras).

| `prop` | Scheme | Status |
|---|---|---|
| `rk4cr` | 4th-order Runge–Kutta for CAS/RAS | Experimental |
| `vv` | Velocity Verlet, CASCI only | **Currently broken** |

!!! bug "`vv` does not run"
    `vv` takes five arguments; the CAS propagation loop calls the integrator
    with six. `prop='vv'` fails immediately with
    `vv() takes 5 positional arguments but 6 were given`. Use `rk4cr`.

!!! danger "Do not use an `RT_SCF` integrator here"
    The two families share one `INTEGRATORS` dictionary and there is no guard.
    Setting `prop='magnus_interpol'` on an `RT_CAS_RAS` object raises a bare
    `TypeError` about positional arguments rather than a helpful message.

`vv` is a second-order symplectic split-operator integrator implemented for
CASCI only — it is not implemented for CASSCF.

## Output

Unlike `RT_SCF`, which logs through PySCF's logger, `RT_CAS_RAS` writes directly
to the two files named in the constructor:

- `outputName` — time, FCI energy, and electron count per step. The electron
  count should never change; it is your stability check.
- `corrDenName` — AO occupations per step.

The CI Hamiltonian is shifted by the t=0 ground-state energy to improve
numerical stability.
