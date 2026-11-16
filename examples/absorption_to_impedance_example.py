if __name__ == "__main__":

    from tmm.tmm import TMM

    # Read published absorption coefficients from the literature table
    table = TMM.absorption_table(category="carpet")
    columns = ["a_125", "a_250", "a_500", "a_1000", "a_2000", "a_4000"]
    print(table[columns])
    bands = [125, 250, 500, 1000, 2000, 4000]
    alpha = table.loc["A-COX-CARPET-9-UL", columns].tolist()

    # Define the frequency range, resolution and sound incidence - a surface defined from absorption coefficients is
    # locally reacting, so its diffuse absorption is the Paris average from 0 to 90 degrees, whatever diffuse_method and
    # incidence_angle say. The default diffuse_method="field" keeps the impedance in the plot and in the export
    mm = TMM(fmin=50, fmax=5000, df=1, incidence="diffuse", filename="example_absorption_to_impedance")

    # Define the surface from the random-incidence absorption coefficients in octave bands
    mm.impedance_from_absorption(bands, alpha, incidence="random", method="model")

    # Plot and export data
    mm.plot(figsize=(7, 7), plots=["alpha", "z"], save_fig=True, show_fig=False, timestamp=False)
    mm.save2sheet(timestamp=False, n_oct=3)
    mm.save()

    # Define the same surface from the id of the row, with a row of the scattering table
    carpet = TMM(fmin=50, fmax=5000, df=1, incidence="diffuse", filename="example_literature_row")
    carpet.material_from_table("A-COX-CARPET-9-UL", method="model", scattering="S-COX-BATTEN-100A")
    carpet.plot(figsize=(7, 7), plots=["alpha", "scat"], save_fig=True, show_fig=False, timestamp=False)
