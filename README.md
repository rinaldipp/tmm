# Transfer Matrix Method (TMM)
Toolbox for design and prediction of multilayered acoustic treatments. Also contains a material model based on the GRAS database.

## Dependencies
- numpy 
- scipy 
- mpmath
- matplotlib
- pandas
- xlsxwriter
- h5py  
- [pytta](https://github.com/PyTTAmaster/PyTTa)

## Installation
    pip install numpy scipy mpmath matplotlib pandas xlsxwriter
    pip install git+https://github.com/pyttamaster/pytta@development
    pip install git+https://github.com/rinaldipp/tmm.git

## Example - TMM
    from tmm.tmm import TMM

    # Define the frequency range, resolution and sound incidence
    treatment = TMM(fmin=20, fmax=5000, df=1, incidence="diffuse")

    # Define the layers - from top to bottom
    treatment.perforated_panel_layer(t=19, d=8, s=24, method="barrier")
    treatment.porous_layer(model="mac", t=50, sigma=27)
    treatment.air_layer(t=50)

    # Compute, plot and export data
    treatment.compute(rigid_backing=True, show_layers=True)
    treatment.plot(figsize=(7, 5), plots=["alpha"], saveFig=True, filename="example_treatment", timestamp=False)
    treatment.save2sheet(timestamp=False, filename="example_treatment", nthOct=1)
    treatment.save("example_treatment")
    bands, filtered_alpha = treatment.filter_alpha(figsize=(7, 5), plot=True, show=True, nthOct=1, returnValues=True)

## Example - MaterialModel
    from tmm.material_model import MaterialModel

    # Define the frequency range, resolution and project parameters
    mm = MaterialModel(fmin=20, fmax=5000, df=1, c0=343, rho0=1.21)

    # Choose the material
    mm.door()
    mm.plot(figsize=(7, 5), plots=["alpha"], saveFig=True, filename="example_door", timestamp=False)
    mm.save2sheet(timestamp=False, filename="example_door", nthOct=1)
    mm.save("example_door")
    bands, filtered_alpha = mm.filter_alpha(figsize=(7, 5), plot="available", show=True, nthOct=1, returnValues=True)

## Contact
For questions about usage, bugs, licensing, citations and/or contributions contact me at rinaldipp@gmail.com.



