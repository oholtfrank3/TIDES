# Installation

## Requirements

- Python ≥ 3.12
- PySCF ≥ 2.7
- NumPy ≥ 1.26.4

## Quick install

```bash
pip install git+https://github.com/oholtfrank3/TIDES@docs
```

This pulls in PySCF and the remaining Python dependencies automatically.

!!! note "Which version is this?"
    These docs describe the `docs` branch of `oholtfrank3/TIDES`, which includes
    functionality not present in the released upstream version — additional
    [integrators](../user-guide/integrators.md),
    [GPU support](../user-guide/gpu.md),
    [CAS/RAS dynamics](../user-guide/cas-ras.md), and
    [ZORA](../user-guide/relativistic.md#zora). Install from the URL above to
    get everything documented here. The upstream project lives at
    [jskretchmer/TIDES](https://github.com/jskretchmer/TIDES).

## Install an optimized BLAS first

!!! warning "Do this before installing"
    Without an optimized BLAS library, PySCF will find one on its own — often a
    slow one. The performance difference is large, not marginal.

For conda users, the simplest route is to let conda pull in its BLAS-linked
stack:

```bash
conda install scipy
```

See the [PySCF installation guide](https://pyscf.org/install.html) for
alternatives (MKL, OpenBLAS) and platform-specific detail.

## Conda environment

The repository ships an `env.yaml` covering the full stack, including the
optional Hirshfeld module:

```bash
conda env create -f env.yaml
conda activate tides
```

## Optional dependencies

Some features need extra packages. TiDES imports without them and tells you what
is missing.

| Feature | Install |
|---|---|
| [Hirshfeld charges/magnetization](../user-guide/observables.md#hirshfeld-partitioning) | `pip install git+https://github.com/frobnitzem/hirshfeld` |
| Hirshfeld-I (`hirshi_atom_charge`) | HORTON |
| [GPU acceleration](../user-guide/gpu.md) | `pip install cupy-cuda12x gpu4pyscf-cuda12x` |
| Ehrenfest trajectory analysis | `pip install MDAnalysis` |
| Workup / plotting scripts | `pip install matplotlib` |

Without the Hirshfeld module you'll see this at import — a note, not an error:

```
Note: Hirshfeld module not installed. Install with
[pip install git+https://github.com/frobnitzem/hirshfeld]
if you wish to collect hirshfeld observables.
```

## Development install

To modify TiDES itself:

```bash
git clone https://github.com/oholtfrank3/TIDES.git
cd TIDES
git checkout docs
pip install -e .
```

The package lives under `src/tides/`, organized into subpackages:

| Subpackage | Contents |
|---|---|
| `tides.methods` | `RT_SCF`, `RT_Ehrenfest`, `RT_CAS_RAS` |
| `tides.propagators` | Integrators and propagation loops |
| `tides.potentials` | Electric fields, CAPs, static fields |
| `tides.observables` | Observable calculation and output |
| `tides.analysis` | Output parsing and spectra |
| `tides.nuclear` | Nuclei, forces, gradients |
| `tides.utils` | Basis utilities, restarts, GPU wrapper |
| `tides.zora` | ZORA relativistic corrections |

The main classes are re-exported at the top level, so prefer

```python
from tides import RT_SCF, RT_Ehrenfest, ElectricField, MOCAP, static_bfield
```

over importing from submodule paths — it is shorter and survives future
reorganization.

## Verify

```bash
python -c "from tides import RT_SCF; print('ok')"
```

Run this from **outside** the source directory. Inside it, Python may import the
local `src/` tree instead of the installed package, which hides installation
problems.

## Running the tests

```bash
cd tests/h_bfield
python test_h_bfield.py
```

Reference output sits in `output.ref` alongside each test.
