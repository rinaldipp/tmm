if __name__ == "__main__":

    from tmm.tmm import TMM

    # Define the frequency range, resolution and sound incidence
    treatment = TMM(fmin=20, fmax=5000, df=1, incidence="diffuse", incidence_angle=[0, 78, 1],
                    filename="example_hdf5_save_and_load")

    # Define the layers - from top to bottom
    treatment.perforated_panel_layer(t=19, d=8, s=24, method="barrier")
    treatment.porous_layer(model="mac", t=50, sigma=27)
    treatment.air_layer(t=50)

    # Compute, plot and export data
    treatment.compute(show_layers=False, rigid_backing=True)
    treatment.plot(plots=["alpha"], save_fig=False, figsize=(8, 8))
    treatment.save()
    treatment.load(treatment.filename)
    treatment.rebuild()
    treatment.plot(plots=["alpha"], save_fig=False, figsize=(8, 8))
