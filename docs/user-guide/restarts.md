# Restarts & Chkfiles

Real-time trajectories are long, and cluster jobs have walltime limits. TiDES
can checkpoint a propagation and resume it.

Everything needed to restart is small: the current time and the current MO
coefficients. Both live in a chkfile.

## Writing a chkfile

Set `chkfile` when constructing the object:

```python
rt = RT_SCF(mf, timestep=0.2, max_time=500, chkfile='Restart.chkfile')
```

The file is created and updated as the propagation runs, at the same
`frequency` as observables.

!!! note "There is always a chkfile"
    If you don't set one, TiDES warns and defaults to `tides.chk`. To avoid
    surprises, name it explicitly.

## Restarting

Point `chkfile` at an **existing** file and TiDES resumes from it:

```python
rt = RT_SCF(mf, timestep=0.2, max_time=1000, chkfile='Restart.chkfile')
```

On construction, TiDES checks whether the path exists. If it does, it loads the
time and MO coefficients and logs `### Restarting from chkfile ###`. If it does
not, it starts from zero and creates the file. The same call does both — there
is no separate "restart" flag.

!!! warning "The chkfile holds only time and MO coefficients"
    It does **not** store your propagation parameters, integrator choice,
    observables, external fields, or fragments. You must redefine all of them
    in the restart script, and they must be consistent with the original run.
    A field defined with a `center` in the past will not re-fire; an integrator
    swapped mid-trajectory will change the dynamics.

`max_time` is measured against the **absolute** time in the chkfile, not the
time remaining. To run a 500 au job and then extend it to 1000 au total, set
`max_time=1000` in the restart — not `max_time=500`.

## A two-part run

`examples/Chkfile/` is a complete worked example, split into `Part1/` and
`Part2/` with a shared workup. It reproduces `examples/Li_ChargeTransfer/`
exactly, but in two jobs — a useful thing to verify for yourself, since a
restart that changes the physics is worse than no restart at all.

## Ehrenfest restarts

`RT_Ehrenfest` also restores nuclear positions and velocities. The `Nuc` object
is created after the base class constructor runs, so `restart_from_chkfile` is
invoked a second time once it exists.

## What "Propagation Stopped Early" means

At the end of `kernel()`, TiDES compares the current time against `max_time`:

- Reached it → logs `Done`
- Did not → logs `Propagation Stopped Early`

The second message on a job you expected to finish usually means the walltime
expired. The chkfile will be current as of the last write, so resume from it.
