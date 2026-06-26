if __name__ == "__main__":

    from tmm.tmm import TMM

    # Define the frequency range, resolution and sound incidence
    treatment = TMM(fmin=10, fmax=5000, df=1, incidence="diffuse", diffuse_method="field",
                    incidence_angle=[0, 78, 1], filename="example_slotted_resonator")

    # Define the layers - from the incident face to the rear termination
    treatment.slotted_panel_layer(t=19, w=4, s=24, method="barrier", rho=500)
    treatment.porous_layer(model="db", t=50, sigma=27)
    treatment.air_layer(t=50)

    # Compute, plot and export data
    treatment.compute(backing="rigid", show_layers=True)
    treatment.plot(plots=["alpha"], save_fig=True, show_fig=False, figsize=(7, 4))
    treatment.save2sheet(n_oct=3)
    treatment.save()
    bands, filtered_alpha = treatment.filter_alpha(view=False, n_oct=3)
