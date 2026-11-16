if __name__ == "__main__":

    from tmm.tmm import TMM

    # List the stored presets - every table also holds the status and the source of each row
    print(TMM.porous_presets(search="glass_wool", sort_by="sigma")[["sigma", "model", "status"]])
    print(TMM.fabric_presets(search="nonwoven")[["surface_mass", "resistance", "method", "status"]])
    print(TMM.plate_presets(search="plasterboard")[["rho", "E", "nu", "loss_factor", "status"]])

    # Define the frequency range, resolution and sound incidence
    treatment = TMM(fmin=20, fmax=5000, df=1, incidence="diffuse", diffuse_method="field",
                    incidence_angle=[0, 78, 1], filename="example_presets")

    # Define the layers - from the incident face to the rear termination
    treatment.perforated_panel_layer(t=12.5, d=6, s=20, method="barrier")
    treatment.porous_facing_layer(preset="pet_nonwoven_70", in_contact=True)
    treatment.porous_layer(preset="glass_wool_11.9k", t=40)
    treatment.air_layer(t=60)

    # Compute, plot and export data
    treatment.compute(backing="rigid", show_layers=True)
    treatment.plot(plots=["alpha"], save_fig=True, show_fig=False)
    treatment.save2sheet(n_oct=3)
    treatment.save()
    bands, filtered_alpha = treatment.filter_alpha(view=False, n_oct=3)
