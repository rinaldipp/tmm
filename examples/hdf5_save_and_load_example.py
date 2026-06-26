if __name__ == "__main__":

    from tmm.tmm import TMM

    # Define the frequency range, resolution and sound incidence
    treatment = TMM(fmin=20, fmax=5000, df=1, incidence="diffuse", diffuse_method="field",
                    incidence_angle=[0, 78, 1],
                    filename="example_hdf5_save_and_load")

    # Define the layers - from the incident face to the rear termination
    treatment.perforated_panel_layer(t=19, d=8, s=24, method="bessel_ingard", rho=2700)
    treatment.porous_layer(model="db", t=50, sigma=27)
    treatment.air_layer(t=50)

    # Compute, plot and export data
    treatment.compute(backing="rigid", show_layers=False)
    treatment.plot(plots=["alpha"], save_fig=False, show_fig=False, figsize=(8, 8))
    treatment.save()
    treatment.load(treatment.filename)
    treatment.rebuild()
    treatment.plot(plots=["alpha"], save_fig=False, show_fig=False, figsize=(8, 8))
