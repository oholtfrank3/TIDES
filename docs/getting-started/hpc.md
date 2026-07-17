# Running on HPC

Real-time trajectories are long-running, serial-in-time jobs: you cannot
parallelize across timesteps, because each step depends on the last. The
parallelism available to you is *within* each Fock build and matrix operation,
which means threaded BLAS.

## Threading

The key point is that TiDES gets its speed from BLAS, and BLAS threading must be
configured deliberately. Set the threads to your allocation and pin them:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

`MKL_DYNAMIC=FALSE` stops MKL from silently reducing the thread count.
`OMP_PROC_BIND`/`OMP_PLACES` pin threads to cores, which matters on shared
nodes.

Mirror the same settings inside the Python script so the process is configured
before PySCF is imported:

```python
import os
from pyscf import gto, scf, lib
from tides import RT_SCF, ElectricField

n_threads = os.environ.get('SLURM_CPUS_PER_TASK', '2')
os.environ['MKL_NUM_THREADS'] = n_threads
os.environ['OPENBLAS_NUM_THREADS'] = n_threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

lib.num_threads(int(n_threads))
```

`lib.num_threads` is PySCF's own thread control and should agree with the
environment.

## A SLURM script

```bash
#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J TiDES
#SBATCH -o TiDES.slo

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

python my_calculation.py > my_calculation.out
```

A single node with one task is the right shape: TiDES is threaded, not
MPI-parallel. Asking for multiple nodes will not make a trajectory faster.

Full example: `examples/Multithreading_SLURM/`.

## Walltime and restarts

Long trajectories will outlive a walltime limit. Always set a
[chkfile](../user-guide/restarts.md) so a job that gets killed can resume:

```python
rt = RT_SCF(mf, 0.2, 5000, chkfile='Restart.chkfile')
```

If your job ends with `Propagation Stopped Early` in the output, it did not
reach `max_time` — resume from the chkfile rather than starting over.

## Output redirection

TiDES logs through PySCF's logger, which writes to stdout by default. Either
redirect it:

```bash
python my_calculation.py > my_calculation.out
```

or name a file on the object:

```python
rt = RT_SCF(mf, 0.2, 500, filename='my_calculation.out')
```

Note that `filename` opens the file in **append** mode, which is what you want
across restarts but means stale output accumulates if you rerun from scratch.

The `Workup_*.py` scripts and `parse_output` both expect the redirected output
file.

## Choosing a machine

The cost driver is the Fock build. Before requesting a large allocation:

- Check the [integrator](../user-guide/integrators.md) — a self-consistent
  integrator builds several Fock matrices per step but tolerates a much larger
  timestep, and usually wins overall.
- Consider [GPU acceleration](../user-guide/gpu.md) for larger systems.
- Consider density fitting, which PySCF supports and TiDES inherits — see
  `examples/TCNE_ChargeTransfer/Density_Fitted/`.
