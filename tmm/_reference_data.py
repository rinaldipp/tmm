"""
Reference data taken from published sources: the porous material parameter sets of ``porous_layer(preset=...)``, the
validity ranges of the empirical porous models, the angular intensity weightings of ``finite_sample_absorption()``, the
rating tables of ``alpha_w()``, the material constants of ``thin_plate_layer(preset=...)`` and
``membrane_layer(preset=...)``, and the fabric and screen parameter sets of ``porous_facing_layer(preset=...)``. Each
table names its source next to it; ``tmm.tmm`` imports every table.

References
----------
[1] J. F. Allard and N. Atalla, "Propagation of Sound in Porous Media: Modelling Sound Absorbing Materials", 2nd ed.,
    Wiley, 2009.
[2] X. Olny and R. Panneton, "Acoustical determination of the parameters governing thermal dissipation in porous
    media", Journal of the Acoustical Society of America, vol. 123, pp. 814-824, 2008.
[3] N. Kino and T. Ueno, "Experimental determination of the micro- and macrostructural parameters influencing the
    acoustical performance of fibrous media", Applied Acoustics, vol. 68, pp. 1439-1458, 2007.
[4] O. Robin, A. Berry, C. K. Amédin, N. Atalla, O. Doutres and F. Sgard, "Laboratory and in situ sound absorption
    measurement under a synthetized diffuse acoustic field", Building Acoustics, 2019, doi:10.1177/1351010X19870307.
[5] P. H. Mareze, E. Brandão, W. D'A. Fonseca, O. M. Silva and A. Lenzi, "Modeling of acoustic porous material absorber
    using rigid multiple micro-ducts network: Validation of the proposed model", Journal of Sound and Vibration, 2018,
    accepted manuscript, doi:10.1016/j.jsv.2018.11.036.
[6] M. Vercammen and M. Lautenbach, "Will the true absorption coefficient please stand up now", DAGA 2016, Aachen,
    2016.
[7] T. Komatsu, "Improvement of the Delany-Bazley and Miki models for fibrous sound-absorbing materials", Acoustical
    Science and Technology, vol. 29, pp. 121-129, 2008.
[8] D. Oliva and V. Hongisto, "Sound absorption of porous materials - Accuracy of prediction methods", Applied
    Acoustics, vol. 74, pp. 1473-1479, 2013.
[9] M. Garai and F. Pompoli, "A simple empirical model of polyester fibre materials for acoustical applications",
    Applied Acoustics, vol. 66, pp. 1383-1398, 2005.
[10] M. Garai and F. Pompoli, "A European inter-laboratory test of airflow resistivity measurements", Acta Acustica
    united with Acustica, vol. 89, pp. 471-478, 2003.
[11] M. Melnyk et al., "Comparison of methods for determining the airflow resistivity of porous and covering
    materials", Archives of Acoustics, vol. 50, pp. 421-433, 2025.
[12] K. V. Horoshenkov et al., "Reproducibility experiments on measuring acoustical properties of rigid-frame porous
    media (round-robin tests)", Journal of the Acoustical Society of America, vol. 122, pp. 345-353, 2007.
[13] D. A. Bies and C. H. Hansen, "Flow resistance information for acoustical design", Applied Acoustics, vol. 13,
    pp. 357-391, 1980.
[14] I. P. Dunn and W. A. Davern, "Calculation of acoustic impedance of multi-layer absorbers", Applied Acoustics,
    vol. 19, pp. 321-334, 1986.
[15] K. O. Ballagh, "Acoustical properties of wool", Applied Acoustics, vol. 48, pp. 101-120, 1996.
[16] C. Wassilieff, "Sound absorption of wood-based materials", Applied Acoustics, vol. 48, pp. 339-356, 1996.
[17] J. F. Wilby and P. H. White, "An analysis of sound absorbing linings for the interior of the NASA Ames
    80x120-foot wind tunnel", NASA Contractor Report CR-177396, 1985.
[18] I. Lee, "Acoustic characteristics of perforated dissipative and hybrid silencers", PhD dissertation, The Ohio
    State University, 2005.
[19] T. Thydal, F. Pind, C.-H. Jeong and A. P. Engsig-Karup, "Experimental validation and uncertainty quantification
    in wave-based computational room acoustics", Applied Acoustics, vol. 178, 107939, 2021. Cited from the preprint
    submitted to the journal, which has no journal page numbers.
[20] F. C. Sgard, X. Olny, N. Atalla and F. Castel, "On the use of perforations to improve the sound absorption of
    porous materials", Applied Acoustics, vol. 66, pp. 625-651, 2005.
[21] M. E. Delany and E. N. Bazley, "Acoustical properties of fibrous absorbent materials", Applied Acoustics, vol. 3,
    pp. 105-116, 1970.
[22] Y. Miki, "Acoustical properties of porous materials - Modifications of Delany-Bazley models", Journal of the
    Acoustical Society of Japan (E), vol. 11, pp. 19-24, 1990.
[23] Q. Wu, "Empirical relations between acoustical properties and flow resistivity of porous plastic open-cell foam",
    Applied Acoustics, vol. 25, pp. 141-148, 1988.
[24] J.-F. Allard and Y. Champoux, "New empirical equations for sound propagation in rigid frame fibrous materials",
    Journal of the Acoustical Society of America, vol. 91, pp. 3346-3353, 1992.
[25] F. P. Mechel, "Ausweitung der Absorberformel von Delany und Bazley zu tiefen Frequenzen", Acustica, vol. 35,
    pp. 210-213, 1976.
[26] K. Attenborough and I. L. Vér, "Sound-absorbing materials and sound absorbers", in I. L. Vér and L. L. Beranek
    (eds.), "Noise and Vibration Control Engineering: Principles and Applications", 2nd ed., Wiley, 2006, chapter 8.
[27] F. P. Mechel (ed.), "Formulas of Acoustics", 2nd ed., Springer, 2008.
[28] C.-H. Jeong, "Converting Sabine absorption coefficients to random incidence absorption coefficients", Journal of
    the Acoustical Society of America, vol. 133, pp. 3951-3962, 2013.
[29] C.-H. Jeong, "Non-uniform sound intensity distributions when measuring absorption coefficients in reverberation
    chambers using a phased beam tracing", Journal of the Acoustical Society of America, vol. 127, pp. 3560-3568, 2010.
[30] ISO 11654:1997, "Acoustics - Sound absorbers for use in buildings - Rating of sound absorption", International
    Organization for Standardization, 1997.
[31] C. Hopkins, "Sound Insulation", Butterworth-Heinemann, 2007.
[32] D. A. Bies and C. H. Hansen, "Engineering Noise Control: Theory and Practice", 4th ed., Spon Press, 2009.
[33] T. E. Vigran, "Building Acoustics", Taylor & Francis, 2008.
[34] L. Jaouen and F.-X. Bécot, "Acoustical characterization of perforated facings", Journal of the Acoustical Society
    of America, vol. 129, pp. 1400-1406, 2011.
[35] T. Kamisiński, K. Brawata, A. Pilch, J. Rubacha and M. Zastawnik, "Sound diffusers with fabric covering", Archives
    of Acoustics, vol. 37, pp. 317-322, 2012.
[36] M. Ognedal, "AbsTex & Sound absorption in free hanging textiles - a brief description", undated summary of the
    author's master thesis, with measurements made at NTNU.
[37] R. Pieren, "Sound absorption modeling of thin woven fabrics backed by an air cavity", Textile Research Journal,
    vol. 82, pp. 864-874, 2012.
[38] T. Okuzono, N. Shimizu and K. Sakagami, "Predicting absorption characteristics of single-leaf permeable membrane
    absorbers using finite element method in a time domain", Applied Acoustics, vol. 151, pp. 172-182, 2019. Cited from
    the author manuscript, which has no journal page numbers.
[39] X. Tang and S. Zhu, "Sound absorption model of thin woven fabric with tortuosity effects by equivalent circuit
    method", Journal of Physics: Conference Series, vol. 2756, 012004, 2024.
[40] S. Sakamoto, K. Sato, G. Muroi, Y. Nakao, K. Kuboki and N. Taguchi, "Improvement of sound-absorbing dips in
    nonwoven fabric sheet with back air space: division of back air space by additional nonwoven fabric sheet",
    Acoustics, vol. 7, 25, 2025.
[41] P. Rebillard, J.-F. Allard, C. Depollier, P. Guignouard, W. Lauriks, C. Verhaegen and A. Cops, "The effect of a
    porous facing on the impedance and the absorption coefficient of a layer of porous material", Journal of Sound and
    Vibration, vol. 156, pp. 541-555, 1992.

For further information check the function specific documentation.
"""
import hashlib

import numpy as np
import pandas


# Material parameter sets for porous_layer(preset=...), from Allard and Atalla (2009), Olny and Panneton (2008), Kino
# and Ueno (2007), Robin et al. (2019), Mareze et al. (2018), Vercammen and Lautenbach (2016), Komatsu (2008), Oliva and
# Hongisto (2013), Garai and Pompoli (2005 and 2003), Melnyk et al. (2025), Horoshenkov et al. (2007), Bies and Hansen
# (1980), Dunn and Davern (1986), Ballagh (1996), Wassilieff (1996), Wilby and White (1985), Lee (2005), Thydal et al.
# (2021) and Miki (1990).
# Names are a generic material class and the flow resistivity in k*Pa*s/m², to 0.1 k*Pa*s/m² with a trailing ".0"
# dropped; a set whose name would repeat one of the same class takes as many further digits as it needs to differ.
# Product and manufacturer names stay out of the names. Units as in porous_layer: sigma [k*Pa*s/m²], lengths [µm],
# thermal_permeability [m²], bulk_density [kg/m³]. "source" names the table, figure or section and the printed page;
# "description" says what the source measured and how, or how it uses the material. "model" is the model porous_layer
# uses when the call gives none: for the complete sets "jcal" when they include a thermal permeability and "jca"
# otherwise, and for the sigma sets the model of their material class, or the model that their source fitted to the
# sample to choose the flow resistivity or whose coefficients it derived from the sample. "kind" is the model of a
# complete set, or "sigma" for a set that gives the flow resistivity and at most the bulk density, porosity and
# tortuosity. "product" and "manufacturer" are None where the source names none. None in a parameter marks a value the
# source does not give. Rows are ordered by flow resistivity. TMM.porous_presets() lists them.
# "status" says how the acoustic values of a row were obtained. The acoustic values are the flow resistivity or
# resistance, porosity, tortuosity, characteristic lengths and thermal permeability, and for plates the density and the
# elastic constants. A secondary value whose origin the source does not state, such as a thickness, surface mass or bulk
# density, is named in the description and does not change the status. A row is "measured" when its acoustic values were
# measured on the sample, by direct or ultrasonic methods. It is "derived" when none of its acoustic values was fitted
# and at least one was computed by arithmetic from values the source prints, and "fitted" when at least one comes from
# indirect characterization, that is, was extracted by inverting an absorber model against measured surface impedance,
# absorption, or dynamic density and bulk modulus. "manufacturer" marks values that come from the producer of the
# material, as the source states. "nominal" marks representative values for a kind of material rather than for one
# sample, such as handbook tables of typical material constants and values a source calls nominal. "unstated" marks a
# row whose source does not say how the acoustic values of the material or product were obtained, also when the package
# computes one of them by arithmetic from values of that unstated origin, which does not make the row "derived". "range"
# marks plasterboard_800 of plate_presets(), the one row that takes the lower ends of printed ranges.
def _preset(sigma, porosity, tortuosity, viscous_length, thermal_length, bulk_density, source, description,
            thermal_permeability=None, model="jca", *, status, product=None, manufacturer=None):
    return dict(model=model, sigma=sigma, porosity=porosity, tortuosity=tortuosity, viscous_length=viscous_length,
                thermal_length=thermal_length, bulk_density=bulk_density, thermal_permeability=thermal_permeability,
                source=source, description=description, kind=model, status=status, product=product,
                manufacturer=manufacturer)


def _sigma_preset(sigma, bulk_density, model, source, description, *, status, porosity=None, tortuosity=None,
                  product=None, manufacturer=None):
    return dict(model=model, sigma=sigma, porosity=porosity, tortuosity=tortuosity, viscous_length=None,
                thermal_length=None, bulk_density=bulk_density, thermal_permeability=None, source=source,
                description=description, kind="sigma", status=status, product=product, manufacturer=manufacturer)


_ROBIN = ("Density, open porosity and static airflow resistivity measured directly at the Université de Sherbrooke, "
          "the other parameters with the indirect impedance-tube method of Doutres et al. that the source cites.")
_KOMATSU = ("Flow resistivity measured by the direct airflow method of ISO 9053 in a 40 mm tube, on the samples of the "
            "source's impedance-tube measurements. The komatsu model is fitted to these samples.")
_OLIVA = ("Flow resistivity measured after ISO 9053:1991 with a controlled unidirectional airflow, on the sample of "
          "the source's impedance-tube measurements, 63.5 mm in diameter; the source had one sample per configuration.")
_GARAI_2005 = ("Thermally bonded polyester fibre of 18 to 48 µm with both faces smoothed (suffix 2SL). Flow "
               "resistivity measured by the alternating airflow method of EN 29053. The garai_pompoli model and its "
               "density law are fitted to this table.")
_GARAI_2003 = ("Grand mean of ten European laboratories, each of which mounted and measured one sample five times. "
               "Nine used the direct airflow method of ISO 9053 and one the alternating method (Table I, p. 473). The "
               "bulk density is the nominal density of the material, 10 ± 1.5 kg/m³ (section 5, p. 473).")
_MELNYK = ("Flow resistivity measured with static airflow after ISO 9053-1:2018 and evaluated with the source's linear "
           "approximation algorithm (PLA). Table 2 also prints a value from fitting the Miki model to measured "
           "absorption, which the preset does not use.")
_HOROSHENKOV = "Means over the seven laboratories of the test, measured independently of the acoustic tests."
_BIES_HANSEN = ("Flow resistivity measured with the source's flow-resistance apparatus (Fig. 1, p. 361); sample "
                "thickness not given.")
_DUNN = "Flow resistivity measured on the source's flow-rate apparatus with a long sample."
_BALLAGH = ("Flow resistivity and bulk density measured for each sample (p. 102); the source does not name the method. "
            "No model in the package is fitted to wool.")
_WASSILIEFF_TORTUOSITY = ("The source treats tortuosity q**2 as an adjustable parameter deduced from measured "
                          "absorption (p. 348).")
_WILBY = ("The table lists properties assumed for analysis, which the text calls test data or estimates, to be "
          "regarded as only nominal (p. 14).")
_LEE = ("Flow resistivity determined by the producer, Owens Corning, with the method of ASTM C522-87; the source "
        "quotes it.")

POROUS_PRESETS = {
    "polyester_0.9k": _sigma_preset(
        0.922, 12.0, "garai_pompoli", "Garai and Pompoli (2005), Table 1, sample 9, p. 1387",
        "Sample 9, nominal thickness 25 mm, mounted 123 mm thick in the flow-resistivity device. " + _GARAI_2005,
        status="measured", product="Fiberform 62 T 2SL", manufacturer="ORV S.p.A."),
    "sheep_wool_1.2k": _sigma_preset(
        1.2, 12.0, "miki", "Ballagh (1996), Table 1, p. 108",
        "Bonded sheep wool batt, 75 mm, one of the samples of the source's random-incidence measurements. " + _BALLAGH
        + " Below the flow resistivity range of the miki data.", status="measured"),
    "foam_1.3k": _sigma_preset(
        1.34, 32.0, "qunli", "Dunn and Davern (1986), Table 1, foam C, p. 325",
        "Fully reticulated polyurethane foam of nominal 60 cells per 25.4 mm. " + _DUNN + " Below the flow "
        "resistivity range of the qunli data.", status="measured"),
    "wood_wool_1.6k": _sigma_preset(
        1.6, None, "wilson_2015", "Wassilieff (1996), Fig. 3, p. 349",
        "Coarse-grain Woodtex, a commercial wood-wool panel with cement binder, 50 mm (also 25 mm, Fig. 4). The "
        "caption gives flow resistivity, porosity and tortuosity as the parameters of the source's model curve. "
        + _WASSILIEFF_TORTUOSITY + " No density printed.", status="fitted", porosity=0.71, tortuosity=2.8,
        product="Woodtex coarse grain"),
    "polyester_2.1k": _sigma_preset(
        2.059, 20.0, "garai_pompoli", "Garai and Pompoli (2005), Table 1, sample 13, p. 1387",
        "Sample 13, nominal thickness 40 mm, mounted 75 mm thick in the flow-resistivity device. " + _GARAI_2005,
        status="measured", product="Fiberform 62 T 2SL", manufacturer="ORV S.p.A."),
    "foam_2.3k": _preset(
        2.3, 0.960, 1.28, 202.0, 376.0, 60.0, "Olny and Panneton (2008), Table I, p. 818, and Table II, p. 819",
        "Polyurethane foam characterized in an impedance tube (12.7 mm sample). Viscous parameters by the "
        "inversion method of Panneton and Olny (2006), thermal parameters by fitting the Lafarge et al. model; "
        "thermal permeability ±23%, thermal length ±7%.", thermal_permeability=167e-10, model="jcal", status="fitted"),
    "polyester_2.4k": _sigma_preset(
        2.435, 45.0, "garai_pompoli", "Melnyk et al. (2025), Table 2, material 1, p. 430",
        "Polyester filling of acoustic panels ('polyester fabric I' of the source), 30 mm. " + _MELNYK,
        status="measured"),
    "polyester_3k": _sigma_preset(
        2.958, 30.0, "garai_pompoli", "Garai and Pompoli (2005), Table 1, sample 16, p. 1387",
        "Sample 16, nominal thickness 50 mm, mounted 48 mm thick in the flow-resistivity device. " + _GARAI_2005,
        status="measured", product="Fiberform 62 T 2SL", manufacturer="ORV S.p.A."),
    "foam_3.2k": _sigma_preset(
        3.2, 38.0, "qunli", "Dunn and Davern (1986), Table 1, foam D, p. 325",
        "Fully reticulated polyurethane foam of nominal 80 cells per 25.4 mm. " + _DUNN, status="measured"),
    "rock_wool_3.4k": _sigma_preset(
        3.38, 36.5, "miki", "Bies and Hansen (1980), Table 1, p. 366",
        "Rockwool of Mineral Fibre Insulation Pty Ltd, sample type 2, first of three rows. " + _BIES_HANSEN
        + " Below the flow resistivity range of the komatsu data.", status="measured",
        manufacturer="Mineral Fibre Insulation Pty Ltd"),
    "glass_wool_3.4k": _sigma_preset(
        3.4, 12.0, "miki", "Wilby and White (1985), Table 1, p. 15",
        "Fibreglass PF 3350, uncompressed; density printed as 0.75 lb/ft³ and 12.0 kg/m³. " + _WILBY
        + " Below the flow resistivity range of the komatsu data.", status="nominal", porosity=0.9952,
        product="PF 3350", manufacturer="Owens Corning"),
    "polyester_4.1k": _preset(
        4.1, 0.964, 1.0354, 269.0, 541.0, 49.9, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 14 of Kino and Ueno (2007), 12 mm thick, of circular fibres of 1.65 tex (39 µm). Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 1380 "
        "kg/m³.", status="derived"),
    "glass_wool_4.9k": _preset(
        4.86, 0.99, 1.0, 225.0, 388.0, 10.0, "Robin et al. (2019), Table 1, p. 6",
        "Glass wool, 80 mm, one of the materials of the source's diffuse-field absorption measurements. " + _ROBIN,
        status="fitted"),
    "glass_fibre_4.9k": _sigma_preset(
        4.896, 100.0, "miki", "Lee (2005), Table 5.1, p. 76",
        "Texturized strand glass fibre of 24 µm filaments (p. 73), filled to 100 kg/m³. " + _LEE + " Below the flow "
        "resistivity range of the komatsu data.", status="manufacturer", product="Advantex",
        manufacturer="Owens Corning"),
    "carpet_5k": _preset(
        5.0, 0.99, 1.0, 23.0, 28.0, 60.0, "Allard and Atalla (2009), Table 11.7, p. 274",
        "Carpet of a layered system (carpet, impervious screen, fibrous layer) bonded onto a rigid wall (Figure 11.16; "
        "measurements of Brouard et al. 1994). The book models the 3.5 mm carpet as two porous layers, since its fibre "
        "bundles are fixed to an impervious backing at the lower face, and Table 11.7 gives both the same values. The "
        "book gives the values as the parameters of its prediction and does not say how they were obtained.",
        status="unstated"),
    "foam_5k": _preset(
        5.0, 0.98, 1.1, 150.0, 216.0, 33.0, "Allard and Atalla (2009), Table 11.3, p. 266",
        "Plastic foam covered by a sheet of glass wool, bonded onto a rigid wall (Figure 11.6). The book gives the "
        "values as the parameters of its prediction and does not say how they were obtained.", status="unstated"),
    "foam_5.04k": _sigma_preset(
        5.04, 15.0, "qunli", "Melnyk et al. (2025), Table 2, material 5, p. 430",
        "Polyurethane foam, 20 mm. " + _MELNYK, status="measured"),
    "polyester_5.1k": _sigma_preset(
        5.112, 40.0, "garai_pompoli", "Garai and Pompoli (2005), Table 1, sample 22, p. 1387",
        "Sample 22, nominal thickness 40 mm, mounted 39 mm thick in the flow-resistivity device. " + _GARAI_2005,
        status="measured", product="Fiberform 62 T 2SL", manufacturer="ORV S.p.A."),
    "polyester_5.7k": _preset(
        5.7, 0.953, 1.0464, 206.0, 399.0, 64.4, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 15 of Kino and Ueno (2007), 11 mm thick, of circular fibres of 1.65 tex (39 µm). Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 1380 "
        "kg/m³.", status="derived"),
    "glass_wool_6k": _sigma_preset(
        6.0, 20.0, "komatsu", "Komatsu (2008), Table 3, glass wool 01, p. 123",
        "Commercial glass wool, 52.5 mm. " + _KOMATSU, status="measured"),
    "foam_6.6k": _preset(
        6.6, 0.98, 1.03, 200.0, 380.0, 11.2, "Allard and Atalla (2009), Table 11.9, p. 276",
        "Foam bonded onto a 0.6 mm aluminium plate for diffuse-field transmission loss. The book gives the values as "
        "the parameters of its prediction and does not say how they were obtained.", status="unstated"),
    "melamine_7.92k": _preset(
        7.92, 0.98, 1.0, 132.0, 149.0, 6.1, "Robin et al. (2019), Table 1, p. 6",
        "Melamine foam, 25 mm, one of the materials of the source's diffuse-field absorption measurements. " + _ROBIN,
        status="fitted"),
    "melamine_7.922k": _sigma_preset(
        7.922, 9.0, "qunli", "Melnyk et al. (2025), Table 2, material 6, p. 430",
        "Melamine foam ('melamine foam I' of the source), 17 mm. " + _MELNYK, status="measured"),
    "sheep_wool_8k": _sigma_preset(
        8.0, 47.0, "miki", "Ballagh (1996), Table 1, p. 108",
        "Needle-punched sheep wool batt, 23 mm, one of the samples of the source's random-incidence measurements. "
        + _BALLAGH, status="measured"),
    "polyester_8.6k": _sigma_preset(
        8.596, 60.0, "garai_pompoli", "Garai and Pompoli (2005), Table 1, sample 23, p. 1387",
        "Sample 23, nominal thickness 15 mm, mounted 30 mm thick in the flow-resistivity device. " + _GARAI_2005,
        status="measured", product="Fiberform 62 T 2SL", manufacturer="ORV S.p.A."),
    "melamine_8.7k": _sigma_preset(
        8.652, 9.6, "qunli", "Melnyk et al. (2025), Table 2, material 7, p. 430",
        "Melamine foam ('melamine foam II' of the source), 30 mm. " + _MELNYK, status="measured"),
    "rock_wool_8.7k": _sigma_preset(
        8.69, 73.3, "komatsu", "Bies and Hansen (1980), Table 1, p. 366",
        "Rockwool of Mineral Fibre Insulation Pty Ltd, sample type 1, first of three rows. " + _BIES_HANSEN,
        status="measured", manufacturer="Mineral Fibre Insulation Pty Ltd"),
    "glass_wool_9k": _sigma_preset(
        9.0, 32.0, "miki", "Miki (1990), Fig. 4, p. 22",
        "Rigidly backed glass wool, 50 mm, with the normalized surface impedance measured by Suzuki, Yano and "
        "Tachibana (1981), the source's reference 4. The source determined the flow resistivity, 9,000 MKS units, so "
        "that the curves of the miki model provide the best fit for f/sigma above 0.02, or f above 200 Hz (section "
        "4.2). Density printed as 32 kg/m³ without saying how it was obtained.", status="fitted"),
    "glass_wool_9.6k": _sigma_preset(
        9.6, 18.0, "komatsu", "Oliva and Hongisto (2013), Table 1, type 1, p. 1474",
        "Glass wool, 50 mm, a raw wool product mainly used for thermal insulation. " + _OLIVA, status="measured"),
    "foam_9.8k": _preset(
        9.77, 0.96, 2.95, 123.0, 227.0, 29.8, "Robin et al. (2019), Table 1, p. 6",
        "Polyurethane foam, 25 mm, one of the materials of the source's diffuse-field absorption measurements. "
        + _ROBIN, status="fitted"),
    "polyester_9.8k": _preset(
        9.8, 0.964, 1.0328, 152.0, 292.0, 49.8, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 12 of Kino and Ueno (2007), 10.5 mm thick, of circular fibres of 0.66 tex (24.7 µm). "
        "Flow resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the "
        "method of Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density "
        "of 1380 kg/m³.", status="derived"),
    "foam_9.9k": _sigma_preset(
        9.9, 8.8, "qunli", "Horoshenkov et al. (2007), Table I, material B, p. 346",
        "Reticulated foam of the round-robin test, 19.7 mm. " + _HOROSHENKOV + " Spread ±0.8 kPa*s/m² in flow "
        "resistivity and ±0.01 in porosity.", status="measured", porosity=0.98),
    "melamine_10.1k": _sigma_preset(
        10.096, 10.0, "qunli", "Garai and Pompoli (2003), Table II, p. 475, and section 5, p. 473",
        "Melamine foam, 40 mm. " + _GARAI_2003 + " Reproducibility standard deviation 10.1 %.", status="measured",
        product="Basotect", manufacturer="BASF"),
    "rock_wool_10.1k": _sigma_preset(
        10.1, 29.7, "komatsu", "Komatsu (2008), Table 3, rock wool 01, p. 123",
        "Commercial rock wool, 35 mm. " + _KOMATSU, status="measured"),
    "polyester_10.2k": _sigma_preset(
        10.2, 38.0, "garai_pompoli", "Oliva and Hongisto (2013), Table 1, type 8, p. 1474",
        "Polyester wool, 50 mm. " + _OLIVA, status="measured"),
    "melamine_10.5k": _preset(
        10.5011, 0.98, 1.01, 106.0, 150.0, 9.8,
        "Mareze et al. (2018), section 3.2, p. 24, and Table 2, p. 25 (pages of the accepted manuscript)",
        "Melamine foam, 40 mm, measured at the UFSM Acoustic Engineering Laboratory: flow resistivity by the "
        "alternating airflow method of ISO 9053 (combined standard uncertainty 1102.6 N*s/m⁴), tortuosity and "
        "characteristic lengths by 40 kHz ultrasound, porosity by dry and wet mass, density by mass and volume.",
        status="measured"),
    "foam_10.9k": _preset(
        10.9, 0.99, 1.02, 100.0, 130.0, 8.8,
        "Allard and Atalla (2009), Table 11.5, layer 3, p. 271 (also Tables 12.1, 12.4, 12.5)",
        "Foam of the felt-screen-foam system, modelled as limp; the same foam is the limp foam of the plate-foam "
        "systems in chapter 12. The book gives the values as the parameters of its prediction and does not say how "
        "they were obtained.", status="unstated"),
    "melamine_11.5k": _sigma_preset(
        11.491, 10.0, "qunli", "Garai and Pompoli (2003), Table II, p. 475, and section 5, p. 473",
        "Melamine foam, 60 mm. " + _GARAI_2003 + " Reproducibility standard deviation 16.1 %.", status="measured",
        product="Basotect", manufacturer="BASF"),
    "glass_wool_11.7k": _sigma_preset(
        11.7, 21.0, "komatsu", "Horoshenkov et al. (2007), Table I, material C, p. 346",
        "Fibreglass of the round-robin test, 29.0 mm. " + _HOROSHENKOV + " Spread ±1.9 kPa*s/m² in flow resistivity "
        "and ±0.03 in porosity.", status="measured", porosity=0.97),
    "glass_wool_11.9k": _preset(
        11.9, 0.989, 1.0108, 143.0, 302.0, 28.0, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 1 of Kino and Ueno (2007), 25 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³.", status="derived"),
    "foam_12.6k": _preset(
        12.569, 0.99, 1.02, 78.0, 192.0, 8.85, "Allard and Atalla (2009), Table 13.1, p. 328",
        "Foam sample of 0.5 m x 0.5 m x 5.08 cm on a rigid wall, used for the finite-size finite-element example. The "
        "book gives the values as the parameters of its prediction and does not say how they were obtained.",
        status="unstated"),
    "glass_wool_12.9k": _preset(
        12.9, 0.99, 1.01, 82.0, 182.0, 26.5, "Vercammen and Lautenbach (2016), Table 1, p. 3",
        "Glass wool faced with glass fibre on both sides, 100 mm, binder content 7.0 %, density ±1.5 kg/m³. The "
        "source takes the set from an acoustic characterization report of Matelys (Jaouen, Bécot and Chevillotte, "
        "2014). Thydal et al. (2021) name the producer.", thermal_permeability=31e-10, model="jcal",
        status="unstated", product="Industry Modus S", manufacturer="Saint-Gobain Ecophon"),
    "glass_wool_14k": _sigma_preset(
        14.0, 25.6, "komatsu", "Wilby and White (1985), Table 1, p. 15",
        "Fibreglass 701; density printed as 1.6 lb/ft³ and 25.6 kg/m³. " + _WILBY, status="nominal",
        porosity=0.9897, product="701", manufacturer="Owens Corning"),
    "glass_wool_16.8k": _preset(
        16.8, 0.987, 1.0093, 132.0, 237.0, 31.8, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 2 of Kino and Ueno (2007), 25 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³.", status="derived"),
    "glass_fibre_17.4k": _sigma_preset(
        17.378, 200.0, "komatsu", "Lee (2005), Table 5.1, p. 76",
        "Texturized strand glass fibre of 24 µm filaments (p. 73), filled to 200 kg/m³. " + _LEE,
        status="manufacturer",
        product="Advantex", manufacturer="Owens Corning"),
    "polyester_17.4k": _preset(
        17.4, 0.947, 1.0494, 113.0, 218.0, 72.8, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 13 of Kino and Ueno (2007), 10.5 mm thick, of circular fibres of 0.66 tex (24.7 µm). "
        "Flow resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the "
        "method of Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density "
        "of 1380 kg/m³.", status="derived"),
    "glass_wool_18.6k": _sigma_preset(
        18.6, 43.6, "komatsu", "Komatsu (2008), Table 3, glass wool 08, p. 123",
        "Commercial glass wool, 52.0 mm. " + _KOMATSU, status="measured"),
    "mineral_wool_18.6k": _sigma_preset(
        18.603, 50.0, "komatsu", "Melnyk et al. (2025), Table 2, material 3, p. 430",
        "Mineral wool with a fibreglass veil, 30 mm. " + _MELNYK, status="measured"),
    "polyester_19.7k": _preset(
        19.7, 0.974, 1.0206, 125.0, 221.0, 35.9, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 10 of Kino and Ueno (2007), 14 mm thick, of circular fibres of 0.22 tex (14.2 µm). "
        "Flow resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the "
        "method of Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density "
        "of 1380 kg/m³.", status="derived"),
    "glass_wool_21.5k": _preset(
        21.5, 0.983, 1.0124, 106.0, 225.0, 41.7, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 3 of Kino and Ueno (2007), 24.5 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³.", status="derived"),
    "foam_22k": _preset(
        22.0, 0.98, 1.9, 87.0, 146.0, 30.0, "Allard and Atalla (2009), Table 11.6, p. 273",
        "Plastic foam surfaced with an impervious screen, bonded onto a rigid wall (Figure 11.14). The book gives the "
        "values as the parameters of its prediction and does not say how they were obtained.", status="unstated"),
    "foam_22.1k": _preset(
        22.1, 0.97, 2.2, 39.0, 275.0, 24.5, "Allard and Atalla (2009), Table 8.1, material 2, p. 174",
        "Material 2 of the chapter 8 surface-wave and layer-mode examples, a 2 cm layer. The book does not name the "
        "material type. The book gives the values as the parameters of its prediction and does not say how they were "
        "obtained.", status="unstated"),
    "glass_fibre_board_22.2k": _preset(
        22.2, 0.96, 1.0, 57.0, 115.0, 66.0, "Robin et al. (2019), Table 1, p. 6",
        "High-density fibreglass board, 22 mm, one of the materials of the source's diffuse-field absorption "
        "measurements. " + _ROBIN, status="fitted"),
    "felt_23k": _preset(
        23.0, 0.99, 1.4, 64.0, 131.0, 66.0, "Allard and Atalla (2009), Table 11.5, layer 1, p. 271",
        "Felt of a felt-screen-foam system on a rigid wall, modelled as a limp porous material. The book gives the "
        "values as the parameters of its prediction and does not say how they were obtained.", status="unstated"),
    "glass_wool_24.3k": _preset(
        24.3, 0.988, 1.01, 63.0, 137.0, 53.0, "Olny and Panneton (2008), Table I, p. 818, and Table II, p. 819",
        "Glass wool characterized in an impedance tube (12.0 mm sample, frame held with pins). Viscous parameters "
        "by the inversion method of Panneton and Olny (2006), thermal parameters by fitting the Lafarge et al. "
        "model; thermal permeability ±10%, thermal length ±6%.", thermal_permeability=31e-10, model="jcal",
        status="fitted"),
    "soft_fibrous_25k": _preset(
        25.0, 0.98, 1.02, 90.0, 180.0, 30.0, "Allard and Atalla (2009), Table 11.2, p. 254",
        "Soft fibrous material used to compare the rigid and limp frame effective densities (Figure 11.2), whose "
        "effective density Panneton (2007) measured. The book gives the values as the parameters of its prediction "
        "and does not say how they were obtained.", status="unstated"),
    "glass_wool_27k": _sigma_preset(
        27.0, 48.0, "komatsu", "Wilby and White (1985), Table 1, p. 15",
        "Fibreglass 703; density printed as 3.0 lb/ft³ and 48.0 kg/m³. " + _WILBY, status="nominal",
        porosity=0.9808, product="703", manufacturer="Owens Corning"),
    "glass_wool_28k": _sigma_preset(
        28.0, 76.0, "mac", "Oliva and Hongisto (2013), Table 1, type 2, p. 1474",
        "Glass wool, 50 mm, a raw wool product mainly used for thermal insulation. " + _OLIVA + " The source derived "
        "the coefficients of its Method 8, Modified Allard and Champoux (the mac model), by a regression of the "
        "impedance and propagation constant that the Allard and Champoux model gives for this 50 mm sample (Wool 2), "
        "over the 1/27 octave band frequencies from 45 to 11,000 Hz (section 2.3.2.8, p. 1476).", status="measured"),
    "glass_wool_29k": _preset(
        29.0, 0.979, 1.0144, 87.0, 184.0, 52.6, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 4 of Kino and Ueno (2007), 25 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³.", status="derived"),
    "wood_fibre_31k": _sigma_preset(
        31.0, 70.0, "wilson_2015", "Wassilieff (1996), Fig. 5, p. 351",
        "Loose Pinus radiata wood fibres held together without binder, 50 mm. Flow resistivity measured by the "
        "static method of ASTM C 522-87 on the sample of the absorption measurement (p. 346). "
        + _WASSILIEFF_TORTUOSITY, status="fitted", porosity=0.85, tortuosity=1.7),
    "fibrous_layer_33k": _preset(
        33.0, 0.98, 1.1, 50.0, 110.0, 60.0, "Allard and Atalla (2009), Table 11.7, p. 274",
        "Fibrous layer behind the impervious screen of a layered system (carpet, impervious screen, fibrous layer) "
        "bonded onto a rigid wall (Figure 11.16). The book gives the values as the parameters of its prediction and "
        "does not say how they were obtained.", status="unstated"),
    "fibrous_blanket_34k": _preset(
        34.0, 0.98, 1.18, 60.0, 86.0, 41.0, "Allard and Atalla (2009), Table 11.4, layer 1, p. 270",
        "Blanket facing a resistive screen and two foams, the layered material bonded onto a rigid wall "
        "(Figure 11.10). The book gives the values as the parameters of its prediction and does not say how they were "
        "obtained.", status="unstated"),
    "mineral_wool_34k": _preset(
        34.0, 0.95, 1.0, 40.0, 80.0, 90.0, "Allard and Atalla (2009), Table 12.2, p. 293",
        "Mineral wool between a steel plate and a laminated plate in a double wall (Villot et al. 2001). The book "
        "gives the values as the parameters of its prediction and does not say how they were obtained.",
        status="unstated"),
    "glass_wool_40k": _preset(
        40.0, 0.94, 1.06, 56.0, 110.0, 130.0, "Allard and Atalla (2009), Table 11.8, p. 275 (also Table 6.1, p. 124)",
        "Glass wool 'Domisol Coffrage', manufactured by Saint-Gobain Isover (section 6.5.4). Anisotropic; these are "
        "its normal-incidence parameters, used for a glass wool bonded onto a 1 mm aluminium plate. The book gives the "
        "values as the parameters of its prediction, computes the characteristic lengths with its equations 5.29 and "
        "5.30 (p. 123) and does not say how the other values were obtained.",
        status="unstated", product="Domisol Coffrage", manufacturer="Saint-Gobain Isover"),
    "rock_wool_41k": _sigma_preset(
        41.0, 150.0, "miki", "Miki (1990), Fig. 5, p. 23",
        "Rigidly backed rockwool, 50 and 100 mm, with the normalized surface impedance measured by Nakajima and Kawai "
        "(1980), the source's reference 5. The source determined the flow resistivity, 41,000 MKS units, so that the "
        "curves of the miki model provide the best fit over the whole frequency range of measurement, with equal "
        "values from either set of data. Fitted only to the data with f/sigma above 0.02, or f above 800 Hz, it varies "
        "from 38,000 to 40,000 MKS units according to the model and the data set, and 41,000 was adopted as a typical "
        "value. Density printed as 150 kg/m³ without saying how it was obtained.", status="fitted"),
    "rock_wool_46k": _sigma_preset(
        46.0, 78.4, "komatsu", "Komatsu (2008), Table 3, rock wool 07, p. 123",
        "Commercial rock wool, 21.0 mm. " + _KOMATSU, status="measured"),
    "glass_wool_47k": _sigma_preset(
        47.0, None, "komatsu", "Thydal et al. (2021), section 2.1.1, p. 3 of the preprint, which has no journal page",
        "Glass wool absorber, 40 mm. The source gives the producer's mean flow resistivity, printed as '47,000 k Pa "
        "s/m2', with a standard deviation of 12.5 %. No density given.", status="manufacturer",
        product="Akusto Wall A", manufacturer="Saint-Gobain Ecophon"),
    "glass_wool_49k": _preset(
        49.0, 0.968, 1.0295, 57.0, 123.0, 81.2, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 5 of Kino and Ueno (2007), 10 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³.", status="derived"),
    "foam_50k": _preset(
        50.0, 0.98, 1.4, 50.0, 150.0, 25.0, "Allard and Atalla (2009), Table 8.1, material 1, p. 174",
        "Material 1 of the chapter 8 Rayleigh-wave predictions. The book does not name the material type; the "
        "surface-wave measurements of that chapter are on open-cell foam. The book gives the values as the parameters "
        "of its prediction and does not say how they were obtained.", status="unstated"),
    "polyester_51k": _preset(
        51.0, 0.947, 1.0471, 67.0, 128.0, 72.9, "Kino and Ueno (2007), Table 2, p. 1443",
        "Polyester fibre sample 11 of Kino and Ueno (2007), 12.5 mm thick, of circular fibres of 0.22 tex (14.2 µm). "
        "Flow resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the "
        "method of Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density "
        "of 1380 kg/m³.", status="derived"),
    "rock_wool_51.2k": _preset(
        51.2, 0.970, 1.06, 41.0, 59.0, 183.0, "Olny and Panneton (2008), Table I, p. 818, and Table II, p. 819",
        "Rock wool characterized in an impedance tube (11.0 mm sample, frame held with pins). Viscous parameters "
        "by the inversion method of Panneton and Olny (2006), thermal parameters by fitting the Lafarge et al. "
        "model; thermal permeability ±8%, thermal length ±14%. Thermal shape factor 2.84, far from the JCA value "
        "of 1.",
        thermal_permeability=12e-10, model="jcal", status="fitted"),
    "foam_65k": _preset(
        65.0, 0.99, 1.98, 37.0, 120.0, 16.0, "Allard and Atalla (2009), Table 11.4, layer 4, p. 270",
        "Foam against the rigid wall in the blanket-screen-foam layered material (Figure 11.10). The book gives the "
        "values as the parameters of its prediction and does not say how they were obtained.", status="unstated"),
    "rock_wool_69.4k": _sigma_preset(
        69.4, 97.0, "komatsu", "Komatsu (2008), Table 3, rock wool 09, p. 123",
        "Commercial rock wool, 24.0 mm. " + _KOMATSU, status="measured"),
    "glass_wool_69.9k": _preset(
        69.9, 0.960, 1.0358, 48.0, 97.0, 101.2, "Kino and Ueno (2007), Table 1, p. 1443",
        "Glass wool sample 6 of Kino and Ueno (2007), 14 mm thick, of glass fibres of nominal 7 µm diameter. Flow "
        "resistivity measured with a device based on ISO 9053, tortuosity and characteristic lengths by the method of "
        "Leclaire et al. with air and argon. Porosity calculated from the bulk density with a fibre density of 2500 "
        "kg/m³.", status="derived"),
    "glass_wool_72.9k": _sigma_preset(
        72.9, 102.4, "komatsu", "Komatsu (2008), Table 3, glass wool 15, p. 123",
        "Commercial glass wool, 8.0 mm. " + _KOMATSU, status="measured"),
    "foam_87k": _preset(
        87.0, 0.97, 2.52, 36.0, 118.0, 31.0, "Allard and Atalla (2009), Table 11.4, layer 3, p. 270",
        "Foam behind the resistive screen in the blanket-screen-foam layered material (Figure 11.10). The book gives "
        "the values as the parameters of its prediction and does not say how they were obtained.",
        status="unstated"),
    "glass_wool_98k": _sigma_preset(
        98.0, 129.0, "miki", "Oliva and Hongisto (2013), Table 1, type 3, p. 1474",
        "Glass wool, 30 mm, a raw wool product mainly used for thermal insulation. " + _OLIVA + " Above the flow "
        "resistivity range of the komatsu and miki data.", status="measured"),
    "glass_wool_126.2k": _sigma_preset(
        126.154, 125.0, "miki", "Melnyk et al. (2025), Table 2, material 4, p. 430",
        "Glass wool, 20 mm. " + _MELNYK + " Above the flow resistivity range of the komatsu and miki data.",
        status="measured"),
    "rock_wool_135k": _preset(
        135.0, 0.94, 2.1, 49.0, 166.0, 130.0,
        "Allard and Atalla (2009), Table 13.2, p. 337; Sgard et al. (2005), Table 1",
        "Rockwool of perforated (double porosity) rockwool samples (Atalla et al. 2001). Includes the static thermal "
        "permeability. The book gives the values as the parameters of its prediction and does not say how they were "
        "obtained.", thermal_permeability=3.3e-9, model="jcal", status="unstated"),
}
PRESET_PARAMETERS = ("sigma", "porosity", "tortuosity", "viscous_length", "thermal_length", "thermal_permeability",
                     "bulk_density")
# Statuses that a row of POROUS_PRESETS, PLATE_PRESETS or FABRIC_PRESETS may hold, as the "status" comments define them.
PRESET_STATUSES = ("measured", "derived", "fitted", "manufacturer", "nominal", "unstated", "range")
# Kinds that a row of POROUS_PRESETS may hold, the model of a complete set or "sigma".
POROUS_PRESET_KINDS = ("jca", "jcal", "sigma")


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
                      "material": "polyester fibre",
                      "source": "Garai and Pompoli (2005), Table 1, p. 1387; the span includes the 110 kg/m³ sample, "
                                "which the paper leaves out of its flow-resistivity law (eq. 2, p. 1390)"},
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


# Material constants for thin_plate_layer(preset=...) and membrane_layer(preset=...), from Hopkins (2007), Bies and
# Hansen (2009), Mechel (2008) and Vigran (2008). Names are the material and its density in kg/m³. Units as in
# thin_plate_layer: rho [kg/m³], E [GPa], nu [-], loss_factor [-]. The loss factor is the internal loss factor of the
# material for bending waves, since thin_plate_layer applies it to the bending stiffness only. None marks a field the
# source does not give as a single value of that kind: printed as a range, of a mixed or unstated kind, or absent. The
# "derived" rows compute E from a printed wave speed, the relation in the description. "source" names the table and
# printed page; "description" says what the source lists and how. Rows are ordered by density. TMM.plate_presets() lists
# them.
# "status" says how the acoustic values of a row were obtained. The acoustic values are the flow resistivity or
# resistance, porosity, tortuosity, characteristic lengths and thermal permeability, and for plates the density and the
# elastic constants. A secondary value whose origin the source does not state, such as a thickness, surface mass or bulk
# density, is named in the description and does not change the status. A row is "measured" when its acoustic values were
# measured on the sample, by direct or ultrasonic methods. It is "derived" when none of its acoustic values was fitted
# and at least one was computed by arithmetic from values the source prints, and "fitted" when at least one comes from
# indirect characterization, that is, was extracted by inverting an absorber model against measured surface impedance,
# absorption, or dynamic density and bulk modulus. "manufacturer" marks values that come from the producer of the
# material, as the source states. "nominal" marks representative values for a kind of material rather than for one
# sample, such as handbook tables of typical material constants and values a source calls nominal. "unstated" marks a
# row whose source does not say how the acoustic values of the material or product were obtained, also when the package
# computes one of them by arithmetic from values of that unstated origin, which does not make the row "derived". "range"
# marks plasterboard_800 of plate_presets(), the one row that takes the lower ends of printed ranges.
def _plate(rho, E, nu, loss_factor, source, description, *, status):
    return dict(rho=rho, E=E, nu=nu, loss_factor=loss_factor, status=status, source=source, description=description)


_HOPKINS_DERIVATION = ("E is derived from the printed quasi-longitudinal phase velocity c_L as E = rho*c_L**2*(1 - "
                       "nu**2) and reproduces the printed h*f_c of ")
_BIES_NOMINAL = ("Bies and Hansen list representative values compiled from several published sources. The loss factor "
                 "is printed as a range from internal to in-situ, so the preset carries none.")
_BIES_PLASTIC_NU = ("Where the source had no Poisson's ratio it computed one from 3-D wave speeds, and for some "
                    "plastics took the value of a similar material.")

PLATE_PRESETS = {
    "osb_590": _plate(
        590, 3.55, 0.3, 0.01, "Hopkins (2007), Table A2, p. 609",
        "Oriented strand board, row credited to Hopkins. The source notes that the board is usually orthotropic, with "
        "c_L between 2200 and 3500 m/s depending on direction, and prints the effective value c_L,eff = 2570 m/s, so "
        "E is an effective in-plane value, not that of one direction. " + _HOPKINS_DERIVATION + "25.2 m*Hz. "
        "nu and the internal loss factor are marked as estimates.", status="derived"),
    "plasterboard_680": _plate(
        680, 2.03, 0.3, 0.0125, "Hopkins (2007), Table A2, p. 609",
        "Plasterboard of flue gas gypsum and natural gypsum, row credited to Hopkins (1999); c_L = 1810 m/s. "
        + _HOPKINS_DERIVATION + "35.8 m*Hz. nu is marked as an estimate; the internal loss factor for "
        "bending waves is printed without that mark.", status="derived"),
    "plywood_birch_710": _plate(
        710, 9.58, 0.3, 0.016, "Hopkins (2007), Table A2, p. 609",
        "Birch plywood, row credited to Hopkins; c_L = 3850 m/s. The source states no direction for this value, so the "
        "preset uses E as an effective isotropic value. " + _HOPKINS_DERIVATION + "16.8 m*Hz. nu is marked as an "
        "estimate; the internal loss factor for bending waves is printed without that mark.", status="derived"),
    "chipboard_760": _plate(
        760, 3.35, 0.3, 0.01, "Hopkins (2007), Table A2, p. 608",
        "Chipboard, row credited to Hopkins; c_L = 2200 m/s. " + _HOPKINS_DERIVATION + "29.5 m*Hz. nu and "
        "the internal loss factor are marked as estimates.", status="derived"),
    "mdf_760": _plate(
        760, 4.53, 0.3, 0.01, "Hopkins (2007), Table A2, p. 608",
        "Medium density fibreboard, row credited to Hopkins; c_L = 2560 m/s. " + _HOPKINS_DERIVATION + "25.3 m*Hz. "
        "nu and the internal loss factor are marked as estimates.", status="derived"),
    "plasterboard_760": _plate(
        760, 2.1, 0.24, None, "Bies and Hansen (2009), Table B.1, p. 667",
        "Gypsum board, listed with sqrt(E/rho) = 1670 m/s. " + _BIES_NOMINAL, status="nominal"),
    "mdf_770": _plate(
        770, 3.7, None, None, "Bies and Hansen (2009), Table B.1, p. 667",
        "MDF in the wood section of the table, listed with sqrt(E/rho) = 2190 m/s. The table leaves Poisson's ratio "
        "blank for wood products without data. " + _BIES_NOMINAL, status="nominal"),
    "plasterboard_800": _plate(
        800, 4.1, 0.3, 0.01, "Vigran (2008), Table 3.1, p. 88",
        "Gypsum plate (plasterboard) in a table of example material data. The source prints density 800-900 kg/m³, "
        "dynamic E = 4.1 GPa, Poisson's ratio about 0.3 and internal loss factor 10-15e-3; the preset takes the lower "
        "ends of the two ranges.", status="range"),
    "plasterboard_fibre_800": _plate(
        800, 2.94, 0.3, None, "Hopkins (2007), Table A2, p. 609",
        "Plasterboard of gypsum with glass fibre and other additives, row credited to Hopkins; c_L = 2010 m/s. "
        + _HOPKINS_DERIVATION + "32.3 m*Hz. nu is marked as an estimate; the source gives no loss factor.",
        status="derived"),
    "plasterboard_860": _plate(
        860, 1.74, 0.3, 0.0141, "Hopkins (2007), Table A2, p. 609",
        "Plasterboard of natural gypsum, row credited to Hopkins (1999); c_L = 1490 m/s. " + _HOPKINS_DERIVATION
        + "43.5 m*Hz. nu is marked as an estimate; the internal loss factor for bending waves is printed "
        "without that mark. These are the thin_plate_layer defaults.", status="derived"),
    "polypropylene_905": _plate(
        905, None, 0.40, None, "Bies and Hansen (2009), Table B.1, p. 668",
        "Polypropylene. The source prints E only as the range 1.4-2.1 GPa, so the preset carries none; membrane_layer "
        "uses only the density. " + _BIES_PLASTIC_NU + " " + _BIES_NOMINAL, status="nominal"),
    "polyethylene_920": _plate(
        920, 0.4, None, None, "Mechel (2008), section I.8, Table 3, p. 530",
        "Soft polyethylene in a handbook table of typical density and elastic constants of materials, printed with "
        "E = 400 MN/m², f_cr*d = 95.5 Hz*m and loss factor 0.1. The table gives no Poisson's ratio and does not say "
        "which kind of loss factor it lists, so the preset carries none. membrane_layer can use the density for "
        "polyethylene film.",
        status="nominal"),
    "hardboard_1000": _plate(
        1000, 4.0, None, None, "Bies and Hansen (2009), Table B.1, p. 667",
        "Compressed hardboard composite in the wood section of the table, listed with sqrt(E/rho) = 2000 m/s. The "
        "table leaves Poisson's ratio blank for wood products without data. " + _BIES_NOMINAL, status="nominal"),
    "acrylic_1190": _plate(
        1190, 3.5, 0.35, None, "Bies and Hansen (2009), Table B.1, p. 668",
        "Plexiglass (acrylic), listed with sqrt(E/rho) = 1710 m/s. " + _BIES_PLASTIC_NU + " " + _BIES_NOMINAL,
        status="nominal"),
    "polycarbonate_1200": _plate(
        1200, 2.3, 0.35, None, "Bies and Hansen (2009), Table B.1, p. 668",
        "Polycarbonate, listed with sqrt(E/rho) = 1380 m/s. " + _BIES_PLASTIC_NU + " " + _BIES_NOMINAL,
        status="nominal"),
    "acrylic_1250": _plate(
        1250, 6.28, 0.3, None, "Hopkins (2007), Table A2, p. 609",
        "Perspex (plexiglass), row credited to Hopkins; c_L = 2350 m/s. " + _HOPKINS_DERIVATION + "27.6 "
        "m*Hz. nu is marked as an estimate; the source gives no loss factor. E is well above the 3.5 GPa of "
        "acrylic_1190; the sources do not say whether their moduli are static or dynamic.", status="derived"),
    "polyester_1310": _plate(
        1310, 2.3, 0.40, None, "Bies and Hansen (2009), Table B.1, p. 668",
        "Thermoplastic polyester, listed with sqrt(E/rho) = 1320 m/s; membrane_layer can use the density for polyester "
        "film. " + _BIES_PLASTIC_NU + " " + _BIES_NOMINAL, status="nominal"),
    "pvc_1400": _plate(
        1400, 2.8, 0.40, None, "Bies and Hansen (2009), Table B.1, p. 668",
        "PVC, listed with sqrt(E/rho) = 1410 m/s. " + _BIES_PLASTIC_NU + " " + _BIES_NOMINAL, status="nominal"),
    "glass_2500": _plate(
        2500, 63.7, 0.24, None, "Hopkins (2007), Table A2, p. 608",
        "Glass, row credited to Hopkins; c_L = 5200 m/s and Poisson's ratio 0.24, both printed without the estimate "
        "mark. " + _HOPKINS_DERIVATION + "12.5 m*Hz. The internal loss factor is printed as the range "
        "0.003-0.006, so the preset carries none.", status="derived"),
    "aluminium_2700": _plate(
        2700, 70, 0.35, None, "Bies and Hansen (2009), Table B.1, p. 665",
        "Aluminium sheet, listed with sqrt(E/rho) = 5150 m/s. " + _BIES_NOMINAL, status="nominal"),
    "steel_7850": _plate(
        7850, 207, 0.30, None, "Bies and Hansen (2009), Table B.1, p. 666",
        "Mild steel, listed with sqrt(E/rho) = 5130 m/s. " + _BIES_NOMINAL + " The density is the membrane_layer "
        "default.", status="nominal"),
    "stainless_steel_7910": _plate(
        7910, 200, 0.30, None, "Bies and Hansen (2009), Table B.1, p. 666",
        "Stainless steel, grade 302, listed with sqrt(E/rho) = 5030 m/s. " + _BIES_NOMINAL, status="nominal"),
    "lead_11340": _plate(
        11340, 13.8, 0.44, None, "Bies and Hansen (2009), Table B.1, p. 666",
        "Lead sheet, listed with sqrt(E/rho) = 1100 m/s. " + _BIES_NOMINAL, status="nominal"),
}
PLATE_PARAMETERS = ("rho", "E", "nu", "loss_factor")


# Fabric and screen parameter sets for porous_facing_layer(preset=...), from Jaouen and Bécot (2011), Kamisiński et al.
# (2012), Ognedal (undated), Pieren (2012), Okuzono et al. (2019), Tang and Zhu (2024), Melnyk et al. (2025), Sakamoto
# et al. (2025), Rebillard et al. (1992) and Allard and Atalla (2009). Names are a generic material class and the
# specific airflow resistance in Pa*s/m, rounded; product and manufacturer names stay in the descriptions. Units as in
# porous_facing_layer: t [mm], surface_mass [kg/m²], resistance [Pa*s/m]. None marks a value the source does not give,
# and also the porosity and tortuosity of the Tang and Zhu (2024) rows, where the source gives an optical perforation
# ratio and a tortuosity computed from it, which the description gives instead. "method" is the branch
# porous_facing_layer uses when the call gives none: "rebillard" where the source gives the porosity and tortuosity of
# the sheet as model parameters, "pieren" otherwise, including the Tang and Zhu (2024) rows. The arithmetic of the
# package, in the "derived" rows and in an "unstated" row whose inputs have no stated origin, is a resistance as
# resistivity times thickness or a surface mass as density times thickness, with the inputs named in the description; a
# change of metric prefix is not arithmetic. "source" names the table or section and the printed page; "description"
# says what the sample is and how the source obtained each value. Rows are ordered by resistance. TMM.fabric_presets()
# lists them.
# "status" says how the acoustic values of a row were obtained. The acoustic values are the flow resistivity or
# resistance, porosity, tortuosity, characteristic lengths and thermal permeability, and for plates the density and the
# elastic constants. A secondary value whose origin the source does not state, such as a thickness, surface mass or bulk
# density, is named in the description and does not change the status. A row is "measured" when its acoustic values were
# measured on the sample, by direct or ultrasonic methods. It is "derived" when none of its acoustic values was fitted
# and at least one was computed by arithmetic from values the source prints, and "fitted" when at least one comes from
# indirect characterization, that is, was extracted by inverting an absorber model against measured surface impedance,
# absorption, or dynamic density and bulk modulus. "manufacturer" marks values that come from the producer of the
# material, as the source states. "nominal" marks representative values for a kind of material rather than for one
# sample, such as handbook tables of typical material constants and values a source calls nominal. "unstated" marks a
# row whose source does not say how the acoustic values of the material or product were obtained, also when the package
# computes one of them by arithmetic from values of that unstated origin, which does not make the row "derived". "range"
# marks plasterboard_800 of plate_presets(), the one row that takes the lower ends of printed ranges.
def _fabric(t, surface_mass, resistance, porosity, tortuosity, source, description, *, method, status):
    return dict(method=method, t=t, surface_mass=surface_mass, resistance=resistance, porosity=porosity,
                tortuosity=tortuosity, status=status, source=source, description=description)


_OKUZONO = ("Okuzono et al. (2019), Table 3, p. 18 of the author manuscript, a preprint without the journal page",
            "Flow resistance (direct flow method), surface density and thickness measured. The source uses the sheets "
            "as single-leaf permeable membrane absorbers over a 0.1 m air cavity.")
_SAKAMOTO = ("Sakamoto et al. (2025), Table 1, p. 3",
             "Ventilation resistance measured with an air permeability tester (p. 2). Thickness and area density are "
             "listed in the source's table of specifications without saying how they were obtained. The source mounts "
             "the sheets on a honeycomb support frame to suppress their vibration.")
_OGNEDAL = ("Ognedal (undated), Table 3, p. 5",
            "From an undated summary of the author's master thesis, with measurements made at NTNU. Airflow "
            "resistivity, weight and thickness measured; the resistivity method is not stated. The resistance is "
            "derived as airflow resistivity times thickness, the product the source itself uses as the specific "
            "airflow resistance (p. 6).")
_PIEREN = ("Pieren (2012), Table 1, p. 871",
           "Surface mass measured with a precision scale. Resistance fitted by hand to impedance-tube absorption with "
           "0.10 m and 0.15 m air cavities, because the signal-to-noise ratio of the ISO 9053 methods is too low "
           "below 50 Pa*s/m (p. 870). Thickness not reported.")
_TANG = ("Tang and Zhu (2024), Table 1, p. 4",
         "Airflow resistance measured per ISO 9053 and thickness with a digital fabric thickness tester; the source "
         "does not say how the areal density was obtained. Table 1 heads the resistance column \"Airflow resistivity "
         "(Pa s/m)\"; the unit and its use in eq. 1 make it a specific airflow resistance. The perforation ratio comes "
         "from pore sizes read with a fabric density mirror and is an optical open area; the source computes the "
         "tortuosity from it, the hydraulic radius of the pores and the measured thickness with its eq. 9 (p. 3). The "
         "preset uses neither value as porosity or tortuosity.")
_MELNYK_FELT = ("Melnyk et al. (2025), Table 1, p. 429",
                "Resistance = airflow resistivity from the static airflow method of ISO 9053-1, "
                "extrapolated to 0.5 mm/s with the source's linear approximation (PLA) algorithm, times thickness; "
                "surface mass = density times thickness. The source does not say how density and thickness were "
                "obtained. Its inverse method (Miki model fitted to impedance-tube absorption, averaged over air gaps "
                "of 10-200 mm) gives a second resistivity, quoted here, which the row does not use.")

FABRIC_PRESETS = {
    "nonwoven_screen_35": _fabric(
        0.35, 0.06, 35.0, 0.72, 1.02, "Jaouen and Bécot (2011), section III.B, pp. 1404-1405",
        "Non-woven screen B. Resistance as printed (sigma*h), from a resistivity of 99 000 Pa*s/m² measured per ISO "
        "9053 on arrangements of six samples, which the source says may be overestimated (its tube estimate is "
        "87 000). Surface mass printed as about 60 g/m² and thickness as 0.35 mm, without saying how either was "
        "obtained. Porosity and tortuosity deduced from the normal surface impedance with a 200 mm air gap.",
        method="rebillard", status="fitted"),
    "polyester_fabric_40": _fabric(
        None, 0.15, 40.0, None, None, "Kamisiński et al. (2012), section 2.1, p. 318",
        "Thin 100 % polyester fabric commonly used to cover sound diffusers. Resistance fitted: the source compares "
        "the normal-incidence absorption of the fabric, measured in an impedance tube at several distances from a "
        "rigid plate, with its limp-sheet model (eqs. 1-4) and prints r*rho0*c = 40 rayl, read as MKS rayl (Pa*s/m). "
        "Surface mass printed as 0.15 kg/m² (pp. 318 and 320); how it was obtained is not stated. Thickness not "
        "reported.", method="pieren", status="fitted"),
    "cover_textile_42": _fabric(
        1.20, 0.1922, 42.0, None, None, _OGNEDAL[0],
        "Fibertex F2B, a strong textile used as cover for mineral wool in slatted panels. Derived: 35 kPa*s/m² times "
        "1.20 mm. Measured weight 192.2 g/m² (the manufacturer gives 140). " + _OGNEDAL[1],
        method="pieren", status="derived"),
    "woven_55": _fabric(
        None, 0.042, 55.0, None, None, _PIEREN[0], "Synthetic woven fabric A. " + _PIEREN[1],
        method="pieren", status="fitted"),
    "pp_nonwoven_64": _fabric(
        0.39, 0.13, 64.0, None, None, _OKUZONO[0], "Polypropylene non-woven fabric A (#5). " + _OKUZONO[1],
        method="pieren", status="measured"),
    "pet_nonwoven_70": _fabric(
        0.31, 0.08, 70.0, None, None, _OKUZONO[0], "PET non-woven fabric A (#7). " + _OKUZONO[1],
        method="pieren", status="measured"),
    "woven_79": _fabric(
        0.547, 0.247, 79.0, None, None, _TANG[0],
        "Thin woven fabric C, perforation ratio 6.14 % and tortuosity 1.266. " + _TANG[1],
        method="pieren", status="measured"),
    "pet_nonwoven_80": _fabric(
        0.31, 0.08, 80.0, None, None, _OKUZONO[0], "PET non-woven fabric B (#8). " + _OKUZONO[1],
        method="pieren", status="measured"),
    "woven_85": _fabric(
        0.539, 0.232, 85.0, None, None, _TANG[0],
        "Thin woven fabric B, perforation ratio 4.39 % and tortuosity 1.392. " + _TANG[1],
        method="pieren", status="measured"),
    "polyester_felt_92": _fabric(
        2.5, 0.3, 92.16, None, None, _MELNYK_FELT[0],
        "Polyester felt I, used as an acoustic panel covering or furniture upholstery. Resistivity 36 864 Pa*s/m² "
        "(static airflow), density 120 kg/m³, thickness 2.5 mm; inverse-method resistivity 49 309 Pa*s/m². "
        + _MELNYK_FELT[1], method="pieren", status="derived"),
    "woven_114": _fabric(
        0.691, 0.248, 114.0, None, None, _TANG[0],
        "Thin woven fabric A, perforation ratio 2.46 % and tortuosity 1.281. " + _TANG[1],
        method="pieren", status="measured"),
    "woven_155": _fabric(
        None, 0.047, 155.0, None, None, _PIEREN[0], "Synthetic woven fabric B. " + _PIEREN[1],
        method="pieren", status="fitted"),
    "pet_cloth_196": _fabric(
        0.16, 0.06, 196.0, None, None, _OKUZONO[0], "PET cloth A (#1). " + _OKUZONO[1],
        method="pieren", status="measured"),
    "nonwoven_214": _fabric(
        0.58, 0.100, 214.0, None, None, _SAKAMOTO[0], "Nonwoven fabric sheet a (RW2100, Idemitsu Kosan). "
        + _SAKAMOTO[1], method="pieren", status="measured"),
    "pp_nonwoven_221": _fabric(
        0.59, 0.24, 221.0, None, None, _OKUZONO[0], "Polypropylene non-woven fabric B (#6). " + _OKUZONO[1],
        method="pieren", status="measured"),
    "woven_260": _fabric(
        None, 0.143, 260.0, None, None, _PIEREN[0], "Synthetic woven fabric C. " + _PIEREN[1],
        method="pieren", status="fitted"),
    "nonwoven_336": _fabric(
        0.4, 0.103, 336.0, None, None, _SAKAMOTO[0], "Nonwoven fabric sheet b (3A01A, Toyobo). " + _SAKAMOTO[1],
        method="pieren", status="measured"),
    "woven_screen_426": _fabric(
        0.55, 0.445, 426.0, 0.04, 1.15, "Jaouen and Bécot (2011), section III.A, pp. 1403-1404",
        "Woven screen A. Resistance as printed (sigma*h), from a mean resistivity of 775 000 Pa*s/m² measured directly "
        "on single samples. Surface mass printed as about 445 g/m² and thickness as 0.55 mm, without saying how "
        "either was obtained. Porosity and tortuosity deduced from the normal surface impedance with a 200 mm air "
        "gap; the source says the tortuosity should be recomputed when the screen is placed against another porous "
        "material.", method="rebillard", status="fitted"),
    "polyester_felt_427": _fabric(
        3.0, 0.51, 426.567, None, None, _MELNYK_FELT[0],
        "Polyester felt II, with finer fibres than felt I, used as an acoustic panel covering or furniture upholstery. "
        "Resistivity 142 189 Pa*s/m² (static airflow), density 170 kg/m³, thickness 3 mm; inverse-method resistivity "
        "114 913 Pa*s/m² (118 180 in the legend of Fig. 7). " + _MELNYK_FELT[1], method="pieren", status="derived"),
    "nonwoven_440": _fabric(
        0.24, 0.070, 440.0, None, None, _SAKAMOTO[0], "Nonwoven fabric sheet c (3701B, Toyobo). " + _SAKAMOTO[1],
        method="pieren", status="measured"),
    "pet_cloth_462": _fabric(
        0.20, 0.12, 462.0, None, None, _OKUZONO[0], "PET cloth B (#2). " + _OKUZONO[1],
        method="pieren", status="measured"),
    "glass_cloth_495": _fabric(
        0.45, 0.297, 495.0, 0.7, 1.0, "Rebillard et al. (1992), Table 2, p. 551",
        "Cloth of glass fibre made by BROCHIER (France), laid on a plastic foam; the source heads its Table 2 "
        "'Measured parameters for the facing'. Derived: resistance = flow resistivity 1 100 000 Pa*s/m² times "
        "thickness 0.045 cm; surface mass = frame density 660 kg/m³ times thickness. Allard and Atalla (2009), "
        "Table 11.3, p. 266, repeat the set as a glass wool sheet without saying how it was obtained.",
        method="rebillard", status="derived"),
    "nonwoven_555": _fabric(
        0.45, 0.152, 555.0, None, None, _SAKAMOTO[0], "Nonwoven fabric sheet d (3A51AD, Toyobo). " + _SAKAMOTO[1],
        method="pieren", status="measured"),
    "wool_serge_736": _fabric(
        1.43, 0.5481, 736.45, None, None, _OGNEDAL[0],
        "M1 Wool Serge, used as variable absorber. Derived: 515 kPa*s/m² times 1.43 mm. Measured weight 548.1 g/m² "
        "(the manufacturer gives 650). " + _OGNEDAL[1], method="pieren", status="derived"),
    "wool_912": _fabric(
        1.41, 0.4775, 912.27, None, None, _OGNEDAL[0],
        "Ullseviot, a tighter textile used as variable absorber. Derived: 647 kPa*s/m² times 1.41 mm. Measured weight "
        "477.5 g/m² (the manufacturer gives 500). " + _OGNEDAL[1], method="pieren", status="derived"),
    "wool_serge_1005": _fabric(
        1.42, 0.4820, 1005.36, None, None, _OGNEDAL[0],
        "Super Wool Serge, used as variable absorber. Derived: 708 kPa*s/m² times 1.42 mm. Measured weight 482.0 g/m² "
        "(the manufacturer gives 500). " + _OGNEDAL[1], method="pieren", status="derived"),
    "pet_woven_1050": _fabric(
        0.48, 0.20, 1050.0, None, None, _OKUZONO[0], "PET high-density woven fabric (#4). " + _OKUZONO[1],
        method="pieren", status="measured"),
    "glass_cloth_1200": _fabric(
        0.18, 0.21, 1200.0, None, None, _OKUZONO[0], "Glass fiber cloth (#3). " + _OKUZONO[1],
        method="pieren", status="measured"),
    "nonwoven_1255": _fabric(
        0.82, 0.250, 1255.0, None, None, _SAKAMOTO[0], "Nonwoven fabric sheet e (RW2250, Idemitsu Kosan). "
        + _SAKAMOTO[1], method="pieren", status="measured"),
    "velour_1536": _fabric(
        2.30, 0.5454, 1536.4, None, None, _OGNEDAL[0],
        "Velour Banner, a strong, fine textile used as stage curtain. Derived: 668 kPa*s/m² times 2.30 mm. Measured "
        "weight 545.4 g/m² (the manufacturer gives 530). " + _OGNEDAL[1], method="pieren", status="derived"),
    "serge_2066": _fabric(
        2.56, 0.8862, 2065.92, None, None, _OGNEDAL[0],
        "Kilo Serge, used as variable absorber. Derived: 807 kPa*s/m² times 2.56 mm. Measured weight 886.2 g/m² (the "
        "manufacturer gives 1000). " + _OGNEDAL[1], method="pieren", status="derived"),
    "screen_2560": _fabric(
        0.8, 0.1, 2560.0, 0.8, 2.56, "Allard and Atalla (2009), Table 11.4, layer 2, p. 270",
        "Thin layer of high flow resistance formed where a fibrous blanket is glued onto a foam, layer 2 of the "
        "blanket-screen-foam-foam material of Figure 11.10 (measurements of Allard et al. 1987). The book gives the "
        "values as the parameters of its prediction and does not say how they were obtained. Resistance = 3.2e6 "
        "Pa*s/m² times 0.8 mm; surface mass = 125 kg/m³ times 0.8 mm.", method="rebillard", status="unstated"),
    "molton_2964": _fabric(
        0.99, 0.3632, 2964.06, None, None, _OGNEDAL[0],
        "Scene Molton, a typical inexpensive stage textile. Derived: 2994 kPa*s/m² times 0.99 mm. Measured weight "
        "363.2 g/m² (the manufacturer gives 300). " + _OGNEDAL[1], method="pieren", status="derived"),
}
FABRIC_PARAMETERS = ("t", "surface_mass", "resistance", "porosity", "tortuosity")


_FULL_DISPLAY = ("display.max_colwidth", None, "display.max_rows", None, "display.max_columns", None)
# Minimum width [em] of a text column whose longest value exceeds the paired length [characters], widest step first.
_TEXT_WIDTH_STEPS = ((200, 32), (60, 20), (20, 12))
# Height above which the HTML table scrolls inside its own container [em].
_TABLE_MAX_HEIGHT = 40
# Background and text colour of the table, set as a pair from one front end: the VS Code editor colours, else the
# JupyterLab layout and content colours, else the browser's system pair. The sticky cells repeat the background.
_TABLE_BACKGROUND = "var(--vscode-editor-background, var(--jp-layout-color0, Canvas))"
_TABLE_FOREGROUND = "var(--vscode-editor-foreground, var(--jp-content-font-color1, CanvasText))"
_TABLE_CSS = """
#{id} {{overflow: auto; max-width: 100%; max-height: {height}em; --tmm-bg: {background};
  background-color: var(--tmm-bg); color: {foreground}; --tmm-stripe: rgba(128, 128, 128, 0.12);}}
#{id} table {{border-collapse: separate; border-spacing: 0; table-layout: auto; margin: 0; color: inherit;}}
#{id} th, #{id} td {{white-space: nowrap; text-align: right; vertical-align: top; padding: 0.25em 0.6em;
  max-width: none;}}
#{id} tbody tr:nth-child(even) > * {{background-image: linear-gradient(var(--tmm-stripe), var(--tmm-stripe));}}
#{id} thead th, #{id} tbody th {{background-color: var(--tmm-bg);}}
#{id} thead th {{position: sticky; top: 0; z-index: 2; border-bottom: 1px solid currentColor;}}
#{id} tbody th, #{id} thead th:first-child {{position: sticky; left: 0; z-index: 1; text-align: left;}}
#{id} thead th:first-child {{z-index: 3;}}
"""
# Placeholder for the table id while the rules are hashed into it.
_TABLE_ID_FIELD = "{id}"


def _shown_labels(index):
    """
    Return ``index`` with every missing label as empty text, level by level for several levels, or ``index`` itself when
    it has none. Dates and durations take the text pandas gives them.
    """
    if isinstance(index, pandas.MultiIndex):
        levels = [index.get_level_values(level) for level in range(index.nlevels)]
        shown = [_shown_labels(level) for level in levels]
        if all(new is old for new, old in zip(shown, levels)):
            return index
        return pandas.MultiIndex.from_arrays(shown, names=index.names)
    missing = np.asarray(index.isna())
    if not missing.any():
        return index
    labels = np.asarray(index.astype(str) if index.dtype.kind in "mM" else index, dtype=object).copy()
    labels[missing] = ""
    return pandas.Index(labels, dtype=object, name=index.name)


class _FullTextFrame(pandas.DataFrame):
    """
    DataFrame that shows every row and column and text columns in full, without changing the global options.

    Missing values (``None``, ``NaN``, ``pandas.NA``, ``NaT``) display as empty cells, row labels included, and keep
    their value in the data. The HTML display scrolls inside its own container with the header row and the index column
    fixed, and long text columns wrap within a minimum width. A frame with several index or column levels, or a named
    column axis, shows as the pandas HTML table, and so does a frame whose own layout raises. With
    ``display.notebook_repr_html`` off there is no HTML display, as in pandas.
    """

    @property
    def _constructor(self):
        return _FullTextFrame

    def _display_frame(self):
        """
        Return a plain copy in which every missing value is a float NaN, which ``na_rep=""`` shows as an empty cell, and
        every missing row label is empty text. Columns of NumPy floats keep their dtype, and the other values keep the
        text pandas gives them.
        """
        frame = pandas.DataFrame(self, copy=True)
        columns = {}
        for position in range(frame.shape[1]):
            column = frame.iloc[:, position]
            columns[position] = column.to_numpy() if isinstance(column.dtype, np.dtype) else column.array
            if isinstance(column.dtype, np.dtype) and column.dtype.kind == "f":
                continue
            missing = column.isna().to_numpy()
            if not missing.any():
                continue
            values = np.full(len(column), np.nan, dtype=object)
            if column.dtype.kind == "c":
                # pandas writes a missing complex value as a number, so the others take their text from pandas here.
                if not missing.all():
                    with pandas.option_context(*_FULL_DISPLAY):
                        texts = column[~missing].to_string(index=False, header=False).splitlines()
                    values[~missing] = [text.strip() for text in texts]
            else:
                # Dates and durations take their column format here, which an object column would lose.
                shown = column.astype(str) if column.dtype.kind in "mM" else column
                values[~missing] = np.asarray(shown, dtype=object)[~missing]
            columns[position] = values
        # A frame built from arrays by position keeps duplicate names and needs no DataFrame.isetitem (pandas 1.5).
        shown_frame = pandas.DataFrame(columns, index=_shown_labels(frame.index))
        shown_frame.columns = frame.columns
        return shown_frame

    def _repr_html_(self):
        if not pandas.get_option("display.notebook_repr_html"):
            return None
        if not (isinstance(self.index, pandas.MultiIndex) or isinstance(self.columns, pandas.MultiIndex)
                or self.columns.name is not None):
            try:
                return self._scrolling_html()
            except Exception:  # noqa: BLE001  (a display never raises; the pandas table below stands in)
                pass
        # Several index or column levels, a named column axis, or a layout that failed: the pandas table, with missing
        # values empty where that step succeeds. If it fails too, there is no HTML and the front end shows the text.
        try:
            frame, na_rep = self._display_frame(), ""
        except Exception:  # noqa: BLE001
            frame, na_rep = pandas.DataFrame(self), "NaN"
        try:
            with pandas.option_context(*_FULL_DISPLAY):
                return frame.to_html(na_rep=na_rep, border=0)
        except Exception:  # noqa: BLE001
            return None

    def _scrolling_html(self):
        """Return the HTML table in its scrolling container, with rules scoped to an id hashed from the rules."""
        rules = [_TABLE_CSS.format(id=_TABLE_ID_FIELD, height=_TABLE_MAX_HEIGHT, background=_TABLE_BACKGROUND,
                                   foreground=_TABLE_FOREGROUND)]
        for position, name in enumerate(self.columns, start=2):  # The index is the first cell of each row.
            column = self[name]
            if isinstance(column, pandas.DataFrame) or pandas.api.types.is_numeric_dtype(column):
                continue
            longest = column.dropna().astype(str).str.len().max()
            width = next((em for length, em in _TEXT_WIDTH_STEPS if longest > length), None)
            if width is not None:
                cells = f"#{_TABLE_ID_FIELD} th:nth-child({position}), #{_TABLE_ID_FIELD} td:nth-child({position})"
                rules.append(f"{cells} {{white-space: normal; min-width: {width}em; text-align: left;}}\n")
        style = "".join(rules)
        # The same rules give the same id, so a notebook's saved output does not change between runs.
        table_id = f"tmm-table-{hashlib.sha256(style.encode('utf-8')).hexdigest()[:32]}"
        # The index name goes to the corner cell, so the header is one row and sticks as one.
        frame = self._display_frame().rename_axis(index=None, columns=self.index.name)
        with pandas.option_context(*_FULL_DISPLAY):
            table = frame.to_html(na_rep="", border=0)
        return f'<div id="{table_id}"><style>{style.replace(_TABLE_ID_FIELD, table_id)}</style>{table}</div>'

    def __repr__(self):
        with pandas.option_context(*_FULL_DISPLAY):
            width = pandas.get_option("display.width") if pandas.get_option("display.expand_frame_repr") else None
            return self._display_frame().to_string(na_rep="", line_width=width)
