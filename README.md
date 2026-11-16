# Transfer Matrix Method (TMM)

TMM is a Python toolbox for building and evaluating multilayer acoustic
treatments with transfer-matrix methods. It supports porous layers, air
cavities, membranes, thin plates, perforated panels, slotted panels,
rectangular-slot panels, porous facings, and material-model helpers based on
the included database.

The package can compute normal, angle, and diffuse-incidence absorption and
impedance, plot common treatment quantities, save/load HDF5 checkpoints, and
export XLSX or CSV result files. It also includes presets for porous materials,
plates, and fabrics, tables of published absorption and scattering
coefficients, and a conversion from band absorption coefficients to a surface
impedance.

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
`matplotlib`, `pandas`, `xlsxwriter`, and `h5py`.

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

## Presets and Literature Data

Porous layers, thin plates, membranes, and porous facings accept a `preset`
name in place of material values, and a membrane takes its density from a plate
preset. The preset tables and the tables of published coefficients are pandas
DataFrames, and every row names its source.

```python
from tmm.tmm import TMM

# List the presets, with optional filters and row order.
porous = TMM.porous_presets(model="jca", sort_by="porosity")
plates = TMM.plate_presets(search="plasterboard")
fabrics = TMM.fabric_presets(status="measured")

# List the published absorption and scattering coefficients.
absorption = TMM.absorption_table(category="carpet")
scattering = TMM.scattering_table()

# Use presets in place of material values.
treatment = TMM(fmin=20, fmax=5000, df=1, incidence="diffuse",
                diffuse_method="field", incidence_angle=[0, 78, 1])
treatment.perforated_panel_layer(t=12.5, d=6, s=20)
treatment.porous_facing_layer(preset="pet_nonwoven_70", in_contact=True)
treatment.porous_layer(preset="glass_wool_11.9k", t=40)
treatment.air_layer(t=60)
treatment.compute(backing="rigid")

# Define a surface from band absorption coefficients, given as values or
# as a row of the absorption table.
surface = TMM(fmin=50, fmax=5000, df=1, incidence="diffuse")
bands = [125, 250, 500, 1000, 2000, 4000]
alpha = [0.08, 0.08, 0.30, 0.60, 0.75, 0.80]
surface.impedance_from_absorption(bands, alpha)
surface.material_from_table("A-COX-CARPET-9-UL")
```

`impedance_from_absorption()` and `material_from_table()` define the whole
surface, as `material_model()` does, so they need no layers and no `compute()`
call. The surface is locally reacting, so its diffuse absorption is the Paris
average from 0 to 90 degrees, whatever `diffuse_method` and `incidence_angle`
say.

## Other Tools

- `alpha_w()` returns the weighted absorption coefficient of ISO 11654 with
  its shape indicators and absorption class.
- `finite_sample_absorption()` returns the diffuse-field absorption of a
  finite rectangular sample, and `plot_finite_sample_absorption()` compares
  its variants.
- `TMM.flow_resistivity()` estimates the flow resistivity of a fibrous
  material from its bulk density and fibre diameter.
- `method="conical"` on `perforated_panel_layer()` and `slotted_panel_layer()`
  models conical holes and wedge-shaped slots, with the exit diameter `d_exit`
  or the exit width `w_exit`.
- `causality_check()` compares the treatment depth with the minimum thickness
  that causality allows for its absorption, `causality_limit()` solves the
  same bound for a flat absorption target, and `check_passivity()` reports
  whether the absorption stays between 0 and 1.

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



