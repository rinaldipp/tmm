# Transfer Matrix Method (TMM)

TMM is a Python toolbox for building and evaluating multilayer acoustic
treatments with transfer-matrix methods. It supports porous layers, air
cavities, membranes, circular perforated panels, long-slot panels, porous
facings, and material-model helpers based on the included database.

The package can compute normal, angle, and diffuse-incidence absorption and
impedance, plot common treatment quantities, save/load HDF5 checkpoints, and
export XLSX or CSV result files.

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/rinaldipp/tmm.git
```

For local development, clone the repository and install it in editable mode:

```bash
pip install -e .
```

Runtime dependencies are declared by the package and include `numpy`, `scipy`,
`matplotlib`, `pandas`, `mpmath`, `xlsxwriter`, and `h5py`.

## Quick Example

```python
from tmm.tmm import TMM

# Define the frequency range, resolution, and sound incidence.
treatment = TMM(
    fmin=20,
    fmax=5000,
    df=1,
    incidence="diffuse",
    diffuse_method="field",
    incidence_angle=[0, 78, 1],
    filename="example_perforated_resonator",
    display_name="Perforated resonator",
)

# Add layers from the incident face toward the rear termination.
treatment.perforated_panel_layer(t=19, d=8, s=24, method="bessel_ingard", rho=2700)
treatment.porous_layer(model="db", t=50, sigma=27)
treatment.air_layer(t=50)

# Compute, plot, export, and save the treatment.
treatment.compute(backing="rigid", show_layers=True)
treatment.plot(plots=["alpha"], save_fig=True, show_fig=True)
treatment.save2sheet(n_oct=3)
treatment.save()
bands, filtered_alpha = treatment.filter_alpha(view=True, n_oct=3)
```

## Incidence Notes

`incidence="normal"` evaluates normal incidence, `incidence="angle"` evaluates
a single configured angle, and `incidence="diffuse"` evaluates a diffuse-field
approximation. For diffuse incidence, `diffuse_method="field"` computes a
field-incidence impedance and absorption result. `diffuse_method="paris"`
computes only a scalar diffuse absorption coefficient from angular absorption
integration, so it does not produce a diffuse complex impedance.

## Examples

Standalone examples are available in the
[examples folder](https://github.com/rinaldipp/tmm/tree/main/examples).

## References

<a id="1">[1]</a>
R. Petrolli, A. Zorzo and P. D'Antonio,
["Comparison of measurement and prediction for acoustical treatments designed with Transfer Matrix Models"](
https://www.researchgate.net/publication/355668614_Comparison_of_measurement_and_prediction_for_acoustical_treatments_designed_with_Transfer_Matrix_Models/references
), in *Euronoise*, October 2021.

## Contact

For questions about usage, bugs, licensing and/or contributions contact
rinaldipp@gmail.com.



