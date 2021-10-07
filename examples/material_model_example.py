if __name__ == "__main__":

    from tmm.tmm import TMM

    # Define the frequency range, resolution and sound incidence
    mm = TMM(fmin=20, fmax=5000, df=1, incidence="normal", filename="example_window")

    # Define material
    mm.material_model(type="window",
                      params={"sample_rate": 44100,
                              "crossover_frequency": 200,
                              "rho_m": 2500,
                              "d": 0.0067,
                              "area": 5.33,
                              "f_res": 6.66,
                              "smooth": True}
                      )

    # Plot and export data
    mm.plot(figsize=(7, 5), plots=["alpha", "scat"], save_fig=True, timestamp=False, max_mode=None)
    mm.save2sheet(timestamp=False, n_oct=3)
    mm.save()
    bands, filtered_alpha = mm.filter_alpha(figsize=(7, 5), plot=True, show=True, n_oct=3)
