# Transfer Matrix Method (TMM)
Toolbox for design and prediction of multi-layered acoustic treatments.

## Dependencies
- numpy 
- scipy 
- mpmath
- matplotlib
- pandas
- xlsxwriter
- [pytta](https://github.com/PyTTAmaster/PyTTa)

## Installation
    pip install numpy scipy mpmath matplotlib pandas xlsxwriter
    pip install git+https://github.com/pyttamaster/pytta@development

## Example

    from tmm import TMM
    
    # Define the frequency range, resolution and sound incidence
    treatment = TMM(fmin=10, fmax=5000, df=1, incidence='normal')

    # Define the layers - from top to bottom
    treatment.perforated_panel_layer(t=19, d=8, s=24)
    treatment.porous_layer(model='ac', t=50, sigma=27)
    treatment.air_layer(t=50)

    # Compute, plot and export data
    treatment.compute(rigid_backing=True, show_layers=True)
    treatment.plot(figsize=(7, 5), plots=['alpha'], saveFig=True, filename='example_treatment')
    treatment.save2sheet(timestamp=False, filename='example_treatment', nthOct=1)
    treatment.filter_alpha(figsize=(7, 5), plot='available', show=True, nthOct=1)

## Contact
For questions about usage, bugs, licensing and/or contributions contact me at [rinaldipp@gmail.com](rinaldipp@gmail.com).



