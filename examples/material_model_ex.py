if __name__ == "__main__":

    from tmm.material_model import MaterialModel

    # Define the frequency range, resolution and project parameters
    mm = MaterialModel(fmin=20, fmax=5000, df=1, c0=343, rho0=1.21)

    # Choose the material
    mm.door()
    mm.plot(figsize=(7, 5), plots=["alpha"], saveFig=True, filename="example_door", timestamp=False)
    mm.save2sheet(timestamp=False, filename="example_door", nthOct=1)
    mm.save("example_door")
    bands, filtered_alpha = mm.filter_alpha(figsize=(7, 5), plot="available", show=True, nthOct=1, returnValues=True)

