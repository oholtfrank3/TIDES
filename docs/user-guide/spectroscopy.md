# Spectroscopy

Real-time methods obtain a full absorption spectrum from a single trajectory. A
weak delta-function field excites every dipole-allowed transition at once; the
dipole moment is propagated and Fourier-transformed.

## The recipe

1. Converge a ground-state SCF calculation
2. Create an `RT_SCF` object and enable the `dipole` observable
3. Apply a weak `'delta'` electric field
4. Propagate
5. Fourier-transform the dipole signal with `tides.analysis.rt_spec.abs_spec`

```python
from pyscf import gto, dft
from tides import RT_SCF, ElectricField

mol = gto.M(atom=..., basis='6-31G*')
rks = dft.RKS(mol)
rks.xc = 'B3LYP'
rks.kernel()

rt = RT_SCF(rks, timestep=0.2, max_time=5000)
rt.observables.update(dipole=True)
rt.add_potential(ElectricField('delta', [0.0001, 0.0, 0.0]))
rt.kernel()
```

## The theory

The frequency-dependent polarizability comes from the ratio of the dipole
response to the applied field strength,

$$\alpha_{ii}(\omega) = \frac{\mu_i(\omega)}{\kappa_i}$$

and the absorption spectrum from its imaginary part,

$$S(\omega) = \frac{4\pi\omega}{c}\sum_{i=x,y,z}\frac{1}{3}\,\mathrm{Im}[\alpha_{ii}(\omega)]$$

Two limits follow directly from the propagation parameters:

| Quantity | Set by |
|---|---|
| Frequency resolution $\Delta\omega = 2\pi/t_{\max}$ | Total simulation time |
| Maximum frequency $\omega_{\max} = \pi/\Delta t$ | Timestep |

Longer runs give sharper peaks; smaller timesteps reach higher frequencies. This
is why XAS needs a much smaller timestep than UV-Vis.

## Polarization

`abs_spec` returns the three Cartesian components. For a properly averaged
spectrum, run **three separate calculations** with the field along x, y, and z
and sum them — this is what the paper does for benzene.

Applying all three components at once in a single run is a convenient shortcut
that is often adequate for a quick look:

```python
ElectricField('delta', [0.0001, 0.0001, 0.0001])
```

but it is not equivalent in general.

## Damping

A finite simulation produces spectral ringing. Damping the dipole signal in time

$$\mu'(t) = \mu(t)\,e^{-t/\tau}$$

broadens the lines, with width inversely proportional to $\tau$. `abs_spec`
takes the damping constant as an argument:

```python
from tides.analysis.parse_rt import parse_output
from tides.analysis.rt_spec import abs_spec

result = parse_output('benzene.out')
w, osc_str = abs_spec(result['time'], result['dipole'], kick_str=0.0001, damp=250)
```

The paper uses $\tau = 250$ for most examples, and no damping for the
spin–orbit spectra.

### `abs_spec` options

```python
abs_spec(time, pole, kick_str=1, pad=None, damp=None,
         hann_damp=None, preprocess_zero=True)
```

| Argument | Meaning |
|---|---|
| `time`, `pole` | Time array and multipole signal, e.g. `result['time']`, `result['dipole']` |
| `kick_str` | Field strength $\kappa$ used for the delta kick. **Must match your input** |
| `damp` | Exponential damping constant $\tau$ |
| `hann_damp` | Hann window as `[t0, sigma]`, an alternative to exponential damping |
| `pad` | Zero-pad the signal by this many points, interpolating the frequency grid |
| `preprocess_zero` | Subtract the $t=0$ dipole before transforming. Default `True` |

`abs_spec` returns `(w, osc_str)` with `w` in atomic units — multiply by 27.2114
for eV — and `osc_str` of shape `(nfreq, 3)`, one column per Cartesian
component.

!!! warning "`kick_str` must match the field you applied"
    The polarizability is the response *divided by* the field strength. Passing
    a `kick_str` that differs from the amplitude in your `ElectricField` silently
    rescales the whole spectrum.

## UV-Vis

![Benzene UV-Vis](../assets/paper/fig04_benzene_uvvis.png){ width="480" }
/// caption
UV-Vis linear-absorption spectrum of benzene from RT-TDDFT in TiDES (solid blue)
and LR-TDDFT in PySCF (dashed black). Figure from Rohan *et al.* (2026),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
///

Benzene at B3LYP/6-31G*, delta kick of 0.0001, $t_{\max} = 5000$,
$\Delta t = 0.2$, $\tau = 250$, three field polarizations. The real-time and
linear-response spectra agree closely, as they should.

Inputs: `examples/ExamplesFromOriginalTiDESPaper/UV-Vis/`. Smaller starting
points: `examples/Water_RHF_UV-Vis/` and `examples/Water_RKS_UV-Vis/`.

## X-ray absorption

![CO K-edge XAS](../assets/paper/fig05_co_xas.png){ width="420" }
/// caption
K-edge spectra of (a) carbon and (b) oxygen in CO. TiDES (solid) against
identical RT-TDDFT calculations in NWChem (dashed). Not shifted to match
experiment. Figure from Rohan *et al.* (2026),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
///

Core-level transitions are high-frequency, so $\omega_{\max} = \pi/\Delta t$
forces a much smaller timestep: the paper uses $\Delta t = 0.02$ with a stronger
0.01 kick and $t_{\max} = 2000$, at B3LYP/Sapporo-QZP-2012-diffuse.

Inputs: `examples/ExamplesFromOriginalTiDESPaper/XAS/`.

## Spin-forbidden transitions via SOC

![SOC spectra of Zn, Cd, Hg](../assets/paper/fig06_soc_zn_cd_hg.png){ width="330" }
/// caption
Linear-absorption spectra of (a) Zn, (b) Cd, (c) Hg including spin–orbit
coupling through X2C1e. Figure from Rohan *et al.* (2026),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
///

Because TiDES inherits whatever is in the PySCF Fock matrix, decorating an SCF
object with X2C1e is enough to capture spin-forbidden singlet–triplet
transitions — visible here for Hg at 5.5 eV. See
[Relativistic Methods](relativistic.md).

Inputs: `examples/ExamplesFromOriginalTiDESPaper/X2C1e/`.

## Alternatives to a plain FFT

Padé approximants can accelerate the Fourier analysis, and extrapolation schemes
allow high-resolution spectra from shorter simulations. Neither is required for
the workflow above.
