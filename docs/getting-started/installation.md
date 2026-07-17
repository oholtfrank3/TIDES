# Installation

## Quick install

```bash
pip install git+https://github.com/jskretchmer/TIDES
```

This pulls in PySCF and the remaining Python dependencies automatically.

## Install an optimized BLAS first

!!! warning "Do this before installing"
    Without an optimized BLAS library, PySCF will find one on its own — often a
    slow one. The performance difference is large, not marginal.

For conda users, the simplest route is to let conda pull in its BLAS-linked stack:

```bash
conda install scipy
```

See the [PySCF installation guide](https://pyscf.org/install.html) for
alternatives (MKL, OpenBLAS) and platform-specific detail.

## Development install

To modify TiDES itself:

```bash
git clone https://github.com/jskretchmer/TIDES.git
cd TIDES
pip install -e .
```

The package lives under `src/tides/`; `pip install -e .` makes edits take effect
without reinstalling.

## Verify

```bash
python -c "import tides; print('ok')"
```

## Running the tests

```bash
cd tests/h_bfield
python test_h_bfield.py
```

Reference output is in `output.ref` alongside each test.
