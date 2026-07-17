# First Calculation

This walkthrough runs a real TiDES calculation end to end: the UV-Vis absorption
spectrum of water, reproducing the
[NWChem RT-TDDFT tutorial](https://nwchemgit.github.io/RT-TDDFT.html#absorption-spectrum-of-water).

The physics: a weak delta-function electric field kicks all dipole-allowed
transitions at once. Propagating the density matrix and Fourier-transforming the
resulting dipole autocorrelation gives the full linear absorption spectrum from a
single trajectory.

## The script

```python
import numpy as np
from pyscf import gto, scf, dft
from tides import rt_scf
from tides.rt_vapp import ElectricField

# 1. Build mol
mol = gto.M(
    verbose = 0,
    atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  ''',
    basis='6-31G',
    spin = 0)

# 2-3. Create and run the SCF object
rks = dft.RKS(mol)
rks.xc = 'PBE0'
rks.kernel()

# 4. Wrap it in RT_SCF
rt_scf = rt_scf.RT_SCF(rks, 0.2, 200)

# 5. Declare observables
rt_scf.observables.update(dipole=True)

# 6. Define and add the external potential
delta_field = ElectricField('delta', [0.0001, 0.0001, 0.0001])
rt_scf.add_potential(delta_field)

# 7. Propagate
rt_scf.kernel()
```

Run it:

```bash
python Water_RKS_UV-Vis.py
```

## What each piece does

### The static reference

TiDES takes a converged PySCF SCF object as its starting point. Any reference
works — RHF, UHF, GHF, RKS, UKS, GKS — and the choice propagates through to the
real-time calculation. Here it's restricted Kohn-Sham with PBE0.

`rks.kernel()` must be called before the object is handed to TiDES.

### Propagation parameters

```python
rt_scf.RT_SCF(rks, 0.2, 200)
```

The positional arguments are `(scf_object, timestep, max_time)`, both in atomic
units.

!!! note "Atomic units"
    1 fs ≈ 41.34 au. So a 0.2 au timestep is ~4.8 as, and a 200 au total time is
    ~4.8 fs.

The timestep is fixed for the whole run. Two optional parameters worth knowing:

- `frequency` — how often observables are printed (default 1, i.e. every step)
- `verbosity` — uses PySCF's logger; default 3 ("Note"), which is where most
  output lives

### Observables

```python
rt_scf.observables.update(dipole=True)
```

Nothing is computed unless you ask for it. For a spectrum you need the dipole.
Other observables (Mulliken and Hirshfeld charges, magnetization, energy) are
enabled the same way.

### The delta kick

```python
ElectricField('delta', [0.0001, 0.0001, 0.0001])
```

The list is the field amplitude along x, y, z. Applying all three at once excites
every polarization direction in a single run — convenient for a total absorption
spectrum.

!!! warning "Keep the kick weak"
    The amplitude must stay in the linear-response regime. Too strong and the
    response is no longer linear and the "spectrum" is meaningless. 0.0001 au is
    a reasonable default.

If you want polarization-resolved spectra instead, run three separate
calculations with the field along each axis — see
[`examples/ExamplesFromOriginalTiDESPaper/UV-Vis/`](https://github.com/jskretchmer/TIDES/tree/main/examples/ExamplesFromOriginalTiDESPaper/UV-Vis),
which does exactly this for benzene.

### Propagating

`rt_scf.kernel()` runs the dynamics and writes observables to the output.

## Working up the result

The raw output is a dipole time series. Converting it to a spectrum means
Fourier-transforming the dipole autocorrelation. Each example ships a
`Workup_*.py` script that does this:

```bash
python Workup_Water_RKS_UV-Vis.py
```

This produces `Water_RKS_UV-Vis_Dipole.png` (the time-domain signal) and
`Water_RKS_UV-Vis_Spectrum.png` (the frequency-domain spectrum). The workup
scripts use `tides.parse_rt` to read the output file — a useful entry point if
you want to do your own analysis.

## Where to go next

| If you want... | Look at |
|---|---|
| Coupled electron–nuclear dynamics | `examples/H2_Ehrenfest/`, `examples/NaCl_Ehrenfest/` |
| Charge transfer | `examples/Li_ChargeTransfer/`, `examples/TCNE_ChargeTransfer/` |
| Core-level spectra | `examples/ExamplesFromOriginalTiDESPaper/XAS/` |
| Metastable / autoionizing states | `examples/Water_MOCAP/` |
| Relativistic effects | `examples/ExamplesFromOriginalTiDESPaper/X2C1e/` |
| Restarting a long run | `examples/Chkfile/` |
| Your own observable or potential | `examples/Custom_Observable/`, `examples/Custom_Potential/` |
| Everything, interactively | `examples/Example_Workbook.ipynb` |
