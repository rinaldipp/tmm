if __name__ == "__main__":

    from tmm.tmm import TMM

    # Define the frequency range, resolution and sound incidence
    treatment = TMM(fmin=20, fmax=5000, df=1, incidence="normal", filename="example_membrane_resonator")

    # Define the layers - from top to bottom
    treatment.membrane_layer(t=3, rho=750)
    treatment.air_layer(t=50)
    treatment.porous_layer(model="mac", t=50, sigma=27)

    # Compute, plot and export data
    treatment.compute(rigid_backing=True, show_layers=True)
    treatment.plot(plots=["alpha"], save_fig=True)
    treatment.save2sheet(n_oct=3)
    treatment.save()
    bands, filtered_alpha = treatment.filter_alpha(view=True, n_oct=3)
