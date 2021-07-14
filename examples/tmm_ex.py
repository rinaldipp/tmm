if __name__ == "__main__":

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

