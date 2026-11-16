if __name__ == "__main__":

    from tmm.tmm import TMM

    # Define the frequency range, resolution and sound incidence
    treatment = TMM(fmin=20, fmax=5000, df=1, incidence="normal", filename="example_membrane_resonator")

    # Define the layers - from the incident face to the rear termination
    treatment.membrane_layer(t=3, preset="mdf_760")
    treatment.air_layer(t=50)
    treatment.porous_layer(preset="rock_wool_41k", t=50)

    # Compute, plot and export data
    treatment.compute(backing="rigid", show_layers=True)
    treatment.plot(plots=["alpha"], save_fig=True, show_fig=False)
    treatment.save2sheet(n_oct=3)
    treatment.save()
    bands, filtered_alpha = treatment.filter_alpha(view=False, n_oct=3)
