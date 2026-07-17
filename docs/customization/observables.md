# Custom Observables

Any quantity you can compute from the RT object can be printed as an observable
**without modifying the TiDES source**. You add two functions from your own input
file.

## The mechanism

Every key in `rt.observables` has a matching key in the protected dictionary
`rt._observables_functions`, holding a two-element list:

```python
rt._observables_functions['OBSERVABLE'] = [calculate_function, print_function]
```

The first function computes the observable; the second prints it. Both take the
RT object as their only argument. Register the pair, switch the observable on,
and TiDES calls them every `frequency` steps.

**The key must be identical in both dictionaries.**

## Template

```python
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

def get_custom_observable(rt_obj):
    # rt_obj is the only argument; read the density via rt_obj.den_ao
    rt_obj._custom_observable = None

def print_custom_observable(rt_obj):
    # Format the output however you like
    rt_obj._log.note(f'HERE IS THE CUSTOM OBSERVABLE: {rt_obj._custom_observable}')

rt_h2o._observables_functions['custom'] = [get_custom_observable, print_custom_observable]
rt_h2o.observables['custom'] = True

rt_h2o.kernel()
```

This is `examples/Blank_Observable/Blank_Observable.py`. A filled-in version is
`examples/Custom_Observable/`.

## Conventions

**Stash results on the object.** The calculate function has no return value —
it should set an attribute (conventionally `_`-prefixed) that the print function
reads. This split exists because computing and printing happen at the same
point but may be gated differently by verbosity.

**Print through the logger, not `print()`.** `rt_obj._log` is PySCF's logger,
so it respects `verbose` and honours the `filename` argument:

| Call | Level | Shown when |
|---|---|---|
| `_log.note(...)` | 3 | Default. Most observables use this |
| `_log.info(...)` | 4 | `verbose>3` |
| `_log.debug(...)` | 5 | `verbose>4` |

Using `_log.info` for bulky per-atom detail while keeping the total at
`_log.note` is the pattern the built-in observables follow.

**What's available on `rt_obj`:**

| Attribute | Contents |
|---|---|
| `den_ao` | Density matrix in the AO basis. `(n,n)` restricted, `(2,n,n)` unrestricted |
| `fock_ao` | Fock matrix in the AO basis, including any external potentials |
| `_fock_orth` | Fock matrix in the orthogonal AO basis |
| `ovlp` | AO overlap matrix |
| `orth`, `orth_inv` | Orthogonalization matrix and its inverse |
| `current_time` | Current time in au |
| `occ` | Occupation vector |
| `_scf` | The underlying PySCF SCF object — `_scf.mol` gets you the molecule |
| `labels` | Atom labels |
| `nmat` | `1` for restricted, `2` for unrestricted |

Handle `nmat` if your observable should work for both restricted and
unrestricted references — the built-in observables do this throughout.

## Reading it back

`parse_output` only knows the built-in keys. To analyze a custom observable,
either parse the output file yourself or write it to a separate file from your
print function.

## Custom observables on CAS objects

The observable machinery is shared: `RT_CAS_RAS` calls the same
`_init_observables` and `get_observables`. The helper argument is named
`rt_obj` rather than `rt_scf` precisely because it may be either class.
