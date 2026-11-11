"""
Reference data taken from published sources: the porous material parameter sets of ``porous_layer(preset=...)``, the
validity ranges of the empirical porous models, the angular intensity weightings of ``finite_sample_absorption()`` and
the rating tables of ``alpha_w()``. Each table names its source next to it; ``tmm.tmm`` imports every table.

References
----------
[1] J. F. Allard and N. Atalla, "Propagation of Sound in Porous Media: Modelling Sound Absorbing Materials", 2nd ed.,
    Wiley, 2009.
[2] X. Olny and R. Panneton, "Acoustical determination of the parameters governing thermal dissipation in porous
    media", Journal of the Acoustical Society of America, vol. 123, pp. 814-824, 2008.
[3] N. Kino and T. Ueno, "Experimental determination of the micro- and macrostructural parameters influencing the
    acoustical performance of fibrous media", Applied Acoustics, vol. 68, pp. 1439-1458, 2007.
[4] M. E. Delany and E. N. Bazley, "Acoustical properties of fibrous absorbent materials", Applied Acoustics, vol. 3,
    pp. 105-116, 1970.
[5] Y. Miki, "Acoustical properties of porous materials - Modifications of Delany-Bazley models", Journal of the
    Acoustical Society of Japan (E), vol. 11, pp. 19-24, 1990.
[6] Q. Wu, "Empirical relations between acoustical properties and flow resistivity of porous plastic open-cell foam",
    Applied Acoustics, vol. 25, pp. 141-148, 1988.
[7] D. Oliva and V. Hongisto, "Sound absorption of porous materials - Accuracy of prediction methods", Applied
    Acoustics, vol. 74, pp. 1473-1479, 2013.
[8] T. Komatsu, "Improvement of the Delany-Bazley and Miki models for fibrous sound-absorbing materials", Acoustical
    Science and Technology, vol. 29, pp. 121-129, 2008.
[9] F. P. Mechel, "Ausweitung der Absorberformel von Delany und Bazley zu tiefen Frequenzen", Acustica, vol. 35,
    pp. 210-213, 1976.
[10] K. Attenborough and I. L. Vér, "Sound-absorbing materials and sound absorbers", in I. L. Vér and L. L. Beranek
    (eds.), "Noise and Vibration Control Engineering: Principles and Applications", 2nd ed., Wiley, 2006, chapter 8.
[11] F. P. Mechel (ed.), "Formulas of Acoustics", 2nd ed., Springer, 2008.
[12] M. Garai and F. Pompoli, "A simple empirical model of polyester fibre materials for acoustical applications",
    Applied Acoustics, vol. 66, pp. 1383-1398, 2005.
[13] C.-H. Jeong, "Non-uniform sound intensity distributions when measuring absorption coefficients in reverberation
    chambers using a phased beam tracing", Journal of the Acoustical Society of America, vol. 127, pp. 3560-3568,
    2010.
[14] C.-H. Jeong, "Converting Sabine absorption coefficients to random incidence absorption coefficients", Journal of
    the Acoustical Society of America, vol. 133, pp. 3951-3962, 2013.
[15] ISO 11654:1997, "Acoustics - Sound absorbers for use in buildings - Rating of sound absorption", International
    Organization for Standardization, 1997.

For further information check the function specific documentation.
"""
import pandas


# Material parameter sets for porous_layer(preset=...), from Allard and Atalla (2009), Olny and Panneton (2008) and Kino
# and Ueno (2007).
# Names are the material and its flow resistivity in kPa*s/m². Units as in porous_layer: sigma [kPa*s/m²], lengths
# [µm], thermal_permeability [m²], bulk_density [kg/m³]. "source" names the table and printed page; "description" says
# how the source uses the material; "model" is the model porous_layer uses when the call gives none, "jcal" for the
# sets that include a thermal permeability. TMM.porous_presets() lists them.
def _preset(sigma, porosity, tortuosity, viscous_length, thermal_length, bulk_density, source, description,
            thermal_permeability=None, model="jca"):
    return dict(model=model, sigma=sigma, porosity=porosity, tortuosity=tortuosity, viscous_length=viscous_length,
                thermal_length=thermal_length, bulk_density=bulk_density, thermal_permeability=thermal_permeability,
                source=source, description=description)


POROUS_PRESETS = {
    "foam_2.3k": _preset(
        2.3, 0.960, 1.28, 202.0, 376.0, 60.0, "Olny and Panneton (2008), Table I, p. 817, and Table II, p. 819",
        "Polyurethane foam characterized in an impedance tube (12.7 mm sample). Viscous parameters by the "
        "inversion method of Panneton and Olny (2006), thermal parameters by fitting the Lafarge et al. model; "
        "thermal permeability ±23%, thermal length ±7%.", thermal_permeability=167e-10, model="jcal"),
    "polyester_4.1k": _preset(
        4.1, 0.964, 1.0354, 269.0, 541.0, 49.9, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 14 of Kino and Ueno (2007), 12 mm thick, of circular fibres of 1.65 tex (39 µm). Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 1380 "
        "kg/m³."),
    "carpet_5k": _preset(
        5.0, 0.99, 1.0, 23.0, 28.0, 60.0, "Table 11.7, p. 274",
        "Carpet of a layered system (carpet, impervious screen, fibrous layer) bonded onto a rigid wall (Figure "
        "11.16; measurements of Brouard et al. 1994). The book models the 3.5 mm carpet as two porous layers, since "
        "its fibre bundles are fixed to an impervious backing at the lower face, and Table 11.7 gives both the same "
        "values."),
    "foam_5k": _preset(
        5.0, 0.98, 1.1, 150.0, 216.0, 33.0, "Table 11.3, p. 266",
        "Plastic foam covered by a sheet of glass wool, bonded onto a rigid wall (Figure 11.6)."),
    "polyester_5.7k": _preset(
        5.7, 0.953, 1.0464, 206.0, 399.0, 64.4, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 15 of Kino and Ueno (2007), 11 mm thick, of circular fibres of 1.65 tex (39 µm). Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 1380 "
        "kg/m³."),
    "foam_6.6k": _preset(
        6.6, 0.98, 1.03, 200.0, 380.0, 11.2, "Table 11.9, p. 276",
        "Foam bonded onto a 0.6 mm aluminium plate for diffuse-field transmission loss."),
    "polyester_9.8k": _preset(
        9.8, 0.964, 1.0328, 152.0, 292.0, 49.8, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 12 of Kino and Ueno (2007), 10.5 mm thick, of circular fibres of 0.66 tex (24.7 µm). "
        "Flow resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the "
        "method of Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density "
        "of 1380 kg/m³."),
    "foam_10.9k": _preset(
        10.9, 0.99, 1.02, 100.0, 130.0, 8.8, "Table 11.5, layer 3, p. 271 (also Tables 12.1, 12.4, 12.5)",
        "Foam of the felt-screen-foam system, modelled as limp; the same foam is the limp foam of the plate-foam "
        "systems in chapter 12."),
    "glass_wool_11.9k": _preset(
        11.9, 0.989, 1.0108, 143.0, 302.0, 28.0, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 1 of Kino and Ueno (2007), 25 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³."),
    "foam_12.6k": _preset(
        12.569, 0.99, 1.02, 78.0, 192.0, 8.85, "Table 13.1, p. 328",
        "Foam sample of 0.5 m x 0.5 m x 5.08 cm on a rigid wall, used for the finite-size finite-element example."),
    "glass_wool_16.8k": _preset(
        16.8, 0.987, 1.0093, 132.0, 237.0, 31.8, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 2 of Kino and Ueno (2007), 25 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³."),
    "polyester_17.4k": _preset(
        17.4, 0.947, 1.0494, 113.0, 218.0, 72.8, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 13 of Kino and Ueno (2007), 10.5 mm thick, of circular fibres of 0.66 tex (24.7 µm). "
        "Flow resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the "
        "method of Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density "
        "of 1380 kg/m³."),
    "polyester_19.7k": _preset(
        19.7, 0.974, 1.0206, 125.0, 221.0, 35.9, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 10 of Kino and Ueno (2007), 14 mm thick, of circular fibres of 0.22 tex (14.2 µm). "
        "Flow resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the "
        "method of Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density "
        "of 1380 kg/m³."),
    "glass_wool_21.5k": _preset(
        21.5, 0.983, 1.0124, 106.0, 225.0, 41.7, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 3 of Kino and Ueno (2007), 24.5 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³."),
    "foam_22k": _preset(
        22.0, 0.98, 1.9, 87.0, 146.0, 30.0, "Table 11.6, p. 273",
        "Plastic foam surfaced with an impervious screen, bonded onto a rigid wall (Figure 11.14)."),
    "foam_22.1k": _preset(
        22.1, 0.97, 2.2, 39.0, 275.0, 24.5, "Table 8.1, material 2, p. 174",
        "Material 2 of the chapter 8 surface-wave and layer-mode examples, a 2 cm layer. The book does not name "
        "the material type."),
    "felt_23k": _preset(
        23.0, 0.99, 1.4, 64.0, 131.0, 66.0, "Table 11.5, layer 1, p. 271",
        "Felt of a felt-screen-foam system on a rigid wall, modelled as a limp porous material."),
    "glass_wool_24.3k": _preset(
        24.3, 0.988, 1.01, 63.0, 137.0, 53.0, "Olny and Panneton (2008), Table I, p. 817, and Table II, p. 819",
        "Glass wool characterized in an impedance tube (12.0 mm sample, frame held with pins). Viscous parameters "
        "by the inversion method of Panneton and Olny (2006), thermal parameters by fitting the Lafarge et al. "
        "model; thermal permeability ±10%, thermal length ±6%.", thermal_permeability=31e-10, model="jcal"),
    "soft_fibrous_25k": _preset(
        25.0, 0.98, 1.02, 90.0, 180.0, 30.0, "Table 11.2, p. 254",
        "Soft fibrous material used to compare the rigid and limp frame effective densities (Figure 11.2), "
        "measured by Panneton (2007)."),
    "glass_wool_29k": _preset(
        29.0, 0.979, 1.0144, 87.0, 184.0, 52.6, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 4 of Kino and Ueno (2007), 25 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³."),
    "fibrous_layer_33k": _preset(
        33.0, 0.98, 1.1, 50.0, 110.0, 60.0, "Table 11.7, p. 274",
        "Fibrous layer behind the impervious screen of a layered system (carpet, impervious screen, fibrous layer) "
        "bonded onto a rigid wall (Figure 11.16)."),
    "fibrous_blanket_34k": _preset(
        34.0, 0.98, 1.18, 60.0, 86.0, 41.0, "Table 11.4, layer 1, p. 270",
        "Blanket facing a resistive screen and two foams, the layered material bonded onto a rigid wall "
        "(Figure 11.10)."),
    "mineral_wool_34k": _preset(
        34.0, 0.95, 1.0, 40.0, 80.0, 90.0, "Table 12.2, p. 293",
        "Mineral wool between a steel plate and a laminated plate in a double wall (Villot et al. 2001)."),
    "glass_wool_40k": _preset(
        40.0, 0.94, 1.06, 56.0, 110.0, 130.0, "Table 11.8, p. 275 (also Table 6.1, p. 124)",
        "Glass wool 'Domisol Coffrage', manufactured by Saint-Gobain Isover (section 6.5.4). Anisotropic; these are "
        "its normal-incidence parameters, used for a glass wool bonded onto a 1 mm aluminium plate."),
    "glass_wool_49k": _preset(
        49.0, 0.968, 1.0295, 57.0, 123.0, 81.2, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 5 of Kino and Ueno (2007), 10 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³."),
    "foam_50k": _preset(
        50.0, 0.98, 1.4, 50.0, 150.0, 25.0, "Table 8.1, material 1, p. 174",
        "Material 1 of the chapter 8 Rayleigh-wave predictions. The book does not name the material type; the "
        "surface-wave measurements of that chapter are on open-cell foam."),
    "polyester_51k": _preset(
        51.0, 0.947, 1.0471, 67.0, 128.0, 72.9, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 11 of Kino and Ueno (2007), 12.5 mm thick, of circular fibres of 0.22 tex (14.2 µm). "
        "Flow resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the "
        "method of Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density "
        "of 1380 kg/m³."),
    "rock_wool_51.2k": _preset(
        51.2, 0.970, 1.06, 41.0, 59.0, 183.0, "Olny and Panneton (2008), Table I, p. 817, and Table II, p. 819",
        "Rock wool characterized in an impedance tube (11.0 mm sample, frame held with pins). Viscous parameters "
        "by the inversion method of Panneton and Olny (2006), thermal parameters by fitting the Lafarge et al. "
        "model; thermal permeability ±8%, thermal length ±14%. Thermal shape factor 2.84, far from the JCA value "
        "of 1.",
        thermal_permeability=12e-10, model="jcal"),
    "foam_65k": _preset(
        65.0, 0.99, 1.98, 37.0, 120.0, 16.0, "Table 11.4, layer 4, p. 270",
        "Foam against the rigid wall in the blanket-screen-foam layered material (Figure 11.10)."),
    "glass_wool_69.9k": _preset(
        69.9, 0.960, 1.0358, 48.0, 97.0, 101.2, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 6 of Kino and Ueno (2007), 14 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³."),
    "foam_87k": _preset(
        87.0, 0.97, 2.52, 36.0, 118.0, 31.0, "Table 11.4, layer 3, p. 270",
        "Foam behind the resistive screen in the blanket-screen-foam layered material (Figure 11.10)."),
    "rock_wool_135k": _preset(
        135.0, 0.94, 2.1, 49.0, 166.0, 130.0, "Table 13.2, p. 337; Sgard et al. (2005), Table 1",
        "Rockwool characterized for perforated (double porosity) rockwool samples (Atalla et al. 2001). Includes "
        "the static thermal permeability.", thermal_permeability=3.3e-9, model="jcal"),
}
PRESET_PARAMETERS = ("sigma", "porosity", "tortuosity", "viscous_length", "thermal_length", "thermal_permeability",
                     "bulk_density")


# Validity ranges of the porous models, as their sources give them. "variable" is the quantity the source bounds:
# "f/sigma" [m³/kg] or "X" = rho0*f/sigma. "basis" is "stated" when the source states the range, "data" when it is only
# the span of the data the model was fitted to. "sigma" [k*Pa*s/m²] and "frequency" [Hz] are the spans of the fitted
# data. None marks an open end or a range the source does not give. The sources of the models absent from the table
# state no range.
POROUS_VALIDITY = {
    "db": {"variable": "f/sigma", "range": (0.01, 1.0), "basis": "stated", "sigma": (2.0, 80.0),
           "frequency": (250.0, 4000.0), "material": "glass fibre and mineral wool",
           "source": "Delany and Bazley (1970), p. 111 (10 < f/sigma <= 1000 in cgs units) and p. 108"},
    "miki": {"variable": "f/sigma", "range": (0.01, 1.0), "basis": "data", "sigma": (2.0, 80.0),
             "frequency": (250.0, 4000.0), "material": "fibrous",
             "source": "Miki (1990), p. 21: regression on the Delany and Bazley data"},
    "qunli": {"variable": None, "range": None, "basis": "data", "sigma": (2.9, 24.3), "frequency": (200.0, 2000.0),
              "material": "open-cell plastic foam of medium flow resistivity",
              "source": "Wu (1988), p. 143 (15 foams) and p. 146"},
    "mac": {"variable": "f/sigma", "range": (45 / 28e3, 11e3 / 28e3), "basis": "data", "sigma": None,
            "frequency": None, "material": "glass wool",
            "source": "Oliva and Hongisto (2013), p. 1476: Method 8, fitted over 45-11000 Hz to Allard and Champoux "
                      "(1992) for one 28 k*Pa*s/m² wool"},
    "komatsu": {"variable": "f/sigma", "range": (100 / 72.9e3, 5000 / 6e3), "basis": "data", "sigma": (6.0, 72.9),
                "frequency": (100.0, 5000.0), "material": "glass wool and rock wool",
                "source": "Komatsu (2008), Table 3 and section 3"},
    "mechel_1976": {"variable": "X", "range": (None, 1.0), "basis": "stated", "sigma": None, "frequency": None,
                    "material": "fibrous",
                    "source": "Mechel (1976), p. 213: power law for 1 <= C <= 60, Rayleigh model for C > 60"},
    "mechel_gf_lowX": {"variable": "X", "range": (None, 0.025), "basis": "stated", "sigma": None, "frequency": None,
                       "material": "glass fibre",
                       "source": "Attenborough and Ver (2006), Table 8.2, p. 242: E <= 0.025"},
    "mechel_gf_highX": {"variable": "X", "range": (0.025, None), "basis": "stated", "sigma": None, "frequency": None,
                        "material": "glass fibre",
                        "source": "Attenborough and Ver (2006), Table 8.2, p. 242: E > 0.025"},
    "mechel_rf_lowX": {"variable": "X", "range": (None, 0.025), "basis": "stated", "sigma": None, "frequency": None,
                       "material": "mineral and basalt wool",
                       "source": "Attenborough and Ver (2006), Table 8.2, p. 242: E <= 0.025"},
    "mechel_rf_highX": {"variable": "X", "range": (0.025, None), "basis": "stated", "sigma": None, "frequency": None,
                        "material": "mineral and basalt wool",
                        "source": "Attenborough and Ver (2006), Table 8.2, p. 242: E > 0.025"},
    "mechel_grundmann": {"variable": "X", "range": (0.003, 0.4), "basis": "stated", "sigma": None, "frequency": None,
                         "material": "mineral fibre (fibre_type=1) and glass fibre (fibre_type=2)",
                         "source": "Mechel et al. (2008), Formulas of Acoustics, section G.11, p. 399"},
    "garai_pompoli": {"variable": None, "range": None, "basis": "data", "sigma": (0.88, 22.699), "frequency": None,
                      "material": "polyester fibre", "source": "Garai and Pompoli (2005), Table 1"},
}

# Angular intensity weightings w(theta) of Jeong (2013), Table II, tabulated from the phased beam-tracing results of
# Jeong (2010). Rows are the 250 Hz to 4 kHz octave bands, columns 0 to 90 degrees in 15-degree steps. Keys follow the
# figure captions of Jeong (2010): Table II labels its w1 block "floor in a rectangular room", but its values are the
# irregular-room floor of Fig. 6b; w2 ("floor in an irregular room") is the rectangular-room floor of Fig. 7; w3
# ("sidewall in a rectangular room") is the irregular-room wall of Fig. 4b. Table II prints 0.11 for w1 at 4 kHz and
# 45 degrees; Fig. 6b shows about 1.1, and 1.11 restores the unit mean over angle of the other rows.
JEONG_WEIGHT_BANDS = (250.0, 500.0, 1000.0, 2000.0, 4000.0)
JEONG_WEIGHT_ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
JEONG_ANGLE_WEIGHTS = {
    "irregular_floor": ((1.63, 1.46, 1.56, 1.46, 0.41, 0.49, 0.0),
                        (1.18, 1.25, 1.14, 1.07, 0.99, 0.82, 0.54),
                        (1.60, 1.58, 1.27, 0.96, 0.84, 0.75, 0.0),
                        (1.76, 1.75, 1.39, 1.00, 0.76, 0.33, 0.0),
                        (1.60, 1.57, 1.40, 1.11, 0.75, 0.40, 0.17)),
    "rectangular_floor": ((1.20, 1.14, 1.04, 1.23, 1.22, 0.80, 0.36),
                          (0.85, 0.69, 1.31, 1.32, 1.25, 0.87, 0.71),
                          (1.38, 1.43, 1.28, 1.23, 0.87, 0.53, 0.26),
                          (1.62, 1.56, 1.43, 1.16, 0.77, 0.42, 0.05),
                          (1.62, 1.57, 1.40, 1.14, 0.84, 0.43, 0.0)),
    "irregular_wall": ((1.25, 1.43, 1.03, 0.99, 1.06, 0.81, 0.43),
                       (1.61, 1.59, 1.43, 1.15, 0.79, 0.42, 0.0),
                       (1.63, 1.57, 1.40, 1.15, 0.82, 0.42, 0.0),
                       (1.64, 1.57, 1.41, 1.15, 0.82, 0.42, 0.0),
                       (1.64, 1.58, 1.41, 1.14, 0.81, 0.41, 0.0)),
}

# ISO 11654:1997 rating. The reference curve of Figure 1 at the 250 Hz to 4 kHz octave bands, and the lowest alpha_w of
# each sound absorption class of Annex B, Table B.1, in hundredths so the 0.05 shifts and the 0.10 limit are exact.
ALPHA_W_REFERENCE = (80, 100, 100, 100, 90)
ALPHA_W_CLASSES = ((90, "A"), (80, "B"), (60, "C"), (30, "D"), (15, "E"))
# Third-octave bands 200 Hz to 5 kHz: exact base-10 mid-band frequencies 1000*10**(x/10), x = -7 to 7.
ALPHA_W_THIRD_OCTAVE_INDICES = range(-7, 8)


class _FullTextFrame(pandas.DataFrame):
    """DataFrame that shows text columns in full when displayed, without changing the global pandas options."""

    @property
    def _constructor(self):
        return _FullTextFrame

    def _repr_html_(self):
        with pandas.option_context("display.max_colwidth", None):
            return super()._repr_html_()

    def __repr__(self):
        with pandas.option_context("display.max_colwidth", None):
            return super().__repr__()
