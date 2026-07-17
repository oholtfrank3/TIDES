# GPU Acceleration

The dominant cost of a real-time calculation is the Fock build, repeated every
step (and several times per step for a self-consistent
[integrator](integrators.md)). TiDES can dispatch that build to a GPU via
[gpu4pyscf](https://github.com/pyscf/gpu4pyscf).

## Installation

```bash
pip install cupy-cuda12x
pip install gpu4pyscf-cuda12x
```

Adjust the CUDA suffix to match your toolkit.

## Usage

Converge a gpu4pyscf SCF object, then wrap it:

```python
from gpu4pyscf import scf as gpu_scf
from tides.utils.rt_gpu import wrap_gpu_mf
from tides import RT_SCF

mf_gpu = gpu_scf.RHF(mol)
mf_gpu.kernel()                    # must be converged before wrapping

mf = wrap_gpu_mf(mf_gpu)           # CPU-side object, GPU-backed Fock builds

rt = RT_SCF(mf, timestep=0.2, max_time=500)
rt.observables.update(energy=True, dipole=True)
rt.kernel()
```

`wrap_gpu_mf` returns a CPU-side PySCF SCF object whose `get_fock` and
`energy_tot` dispatch to the GPU. It is a drop-in replacement for the `scf`
argument to `RT_SCF` or `RT_Ehrenfest` — everything else in your input is
unchanged.

## How it works

Real-time propagation needs Fock builds from **complex** density matrices, but
the GPU kernels require real input. `wrap_gpu_mf` splits a complex density into
its real and imaginary parts, calls the GPU kernel twice, and recombines the
results analytically.

The rest of the calculation — the matrix exponential, observables, output —
stays on the CPU.

## Errors it raises

| Condition | Exception |
|---|---|
| `cupy` or `gpu4pyscf` not installed | `ImportError` |
| Argument is not a recognised gpu4pyscf SCF object | `TypeError` |
| SCF object not converged before wrapping | `RuntimeError` |

The convergence check exists because wrapping an unconverged object silently
produces a meaningless starting state.

!!! note "Whether the GPU helps depends on the system"
    GPU acceleration pays off when the Fock build dominates — larger basis sets
    and larger molecules. For small systems, transfer overhead can outweigh the
    gain. Benchmark your own case.
