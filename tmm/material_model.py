import numpy as np
import os
from matplotlib import ticker, gridspec, style, rcParams
from matplotlib import pyplot as plt
from matplotlib import style
import pandas
import xlsxwriter
from scipy.interpolate import CubicSpline
from scipy.signal import butter, freqz, savgol_filter
import time
import pytta
try:
    from tmm.database.path import path as database_path
except:
    from database.path import path as database_path

outputs = os.getcwd()
style.use('seaborn-colorblind')


class MaterialModel:
    """
    Models for different surfaces in the GRAS database published in the supplemental data of
    "A framework for auralization of boundary element method simulations including source and receiver directivity"
    by Jonathan A. Hargreaves, Luke R. Rendell, and Yiu W. Lam.

    GRAS database: https://depositonce.tu-berlin.de//handle/11303/7506
    Supplemental data: https://asa.scitation.org/doi/suppl/10.1121/1.5096171

    Available materials:
    -------------------
     - Floor
     - Ceiling
     - Door
     - Concrete
     - Plaster
     - MDF
     - Windows
    """

    def __init__(self, fmin=20, fmax=5000, df=0.5, rho0=1.21, c0=343, project_folder=None):

        self.fmin = fmin
        self.fmax = fmax
        self.df = df  # Frequency resolution
        self.freq = np.linspace(self.fmin, self.fmax, int((self.fmax - self.fmin) / self.df) + 1)  # Freqeuency vector
        self.rho0 = rho0  # Air density [kg/m³]
        self.c0 = c0  # Speed of sound [m/s]
        self.z = np.zeros_like(self.freq, dtype='complex')  # Complex impedance
        self.s = None  # Scattering coefficient (if available)
        self.database = database_path()  # Folder path containing the GRAS database files
        self.project_folder = project_folder

    @property
    def z0(self):
        return 1
        # return self.rho0 * self.c0

    @property
    def alpha(self):
        R, alpha = self.reflection_and_absorption_coefficient(self.z)
        return alpha.reshape((len(alpha),))

    @property
    def y(self):
        return 1 / self.z

    def reflection_and_absorption_coefficient(self, zs):
        """
        Calculate reflection coefficient and absorption coefficient for a given surface impedance.

        Parameters
        ----------
        zs : array
            Surface impedance.
        Returns
        -------
        R : array
            Reflection coefficient.
        alpha : array
            Absorption coefficient.
        """
        R = (zs - self.z0) / (zs + self.z0)
        alpha = 1 - np.abs(R) ** 2

        return R, alpha

    def floor(self):
        """
        This is a model of the floor material defined in Scene 9 of the GRAS database. It is a purely real (resistive)
        admittance found from the measured absorption coefficient data using a spline fit.
        """
        # Load the random incidence absorption coefficient data included in the GRAS database:
        csvData = pandas.read_csv(self.database + '_csv' + os.sep + 'mat_scene09_floor.csv', header=None).T
        fMeas = csvData[0]  # Third-octave band center frequencies
        aMeas = csvData[1]  # Third-octave band center absorption coefficients
        sMeas = csvData[2]  # Third-octave band center scattering coefficients

        # Convert to purely real admittance assuming material follows '55 degree rule':
        YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

        # Interpolate to specific frequency list using a spline fit:
        Yf = CubicSpline(fMeas, YsMeas, bc_type='natural')
        Sf = CubicSpline(fMeas, sMeas, bc_type='natural')
        YsInterp = Yf(self.freq)
        SsInterp = Sf(self.freq)

        self.z = 1 / YsInterp
        self.s = SsInterp

    def ceiling(self):
        """
        This is a model of the ceiling material defined in Scene 9 of the GRAS database. It is a purely real (resistive)
        admittance found from the measured absorption coefficient data using a spline fit.
        """
        # Load the random incidence absorption coefficient data included in the GRAS database:
        csvData = pandas.read_csv(self.database + '_csv' + os.sep + 'mat_scene09_ceiling.csv', header=None).T
        fMeas = csvData[0]  # Third-octave band center frequencies
        aMeas = csvData[1]  # Third-octave band center absorption coefficients
        sMeas = csvData[2]  # Third-octave band center scattering coefficients

        # Convert to purely real admittance assuming material follows '55 degree rule':
        YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

        # Interpolate to specific frequency list using a spline fit:
        Yf = CubicSpline(fMeas, YsMeas, bc_type='natural')
        Sf = CubicSpline(fMeas, sMeas, bc_type='natural')
        YsInterp = Yf(self.freq)
        SsInterp = Sf(self.freq)

        self.z = 1 / YsInterp
        self.s = SsInterp

    def concrete(self):
        """
        This is a model of the concrete material defined in Scene 9 of the GRAS database. It is a purely real (resistive)
        admittance found from the measured absorption coefficient data using a spline fit.
        """
        # Load the random incidence absorption coefficient data included in the GRAS database:
        csvData = pandas.read_csv(self.database + '_csv' + os.sep + 'mat_scene09_concrete.csv', header=None).T
        fMeas = csvData[0]  # Third-octave band center frequencies
        aMeas = csvData[1]  # Third-octave band center absorption coefficients
        sMeas = csvData[2]  # Third-octave band center scattering coefficients

        # Convert to purely real admittance assuming material follows '55 degree rule':
        YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

        # Interpolate to specific frequency list using a spline fit:
        Yf = CubicSpline(fMeas, YsMeas, bc_type='natural')
        Sf = CubicSpline(fMeas, sMeas, bc_type='natural')
        YsInterp = Yf(self.freq)
        SsInterp = Sf(self.freq)

        self.z = 1 / YsInterp
        self.s = SsInterp

    def plaster(self):
        """
        This is a model of the plaster material defined in Scene 9 of the GRAS database. It is a purely real (resistive)
        admittance found from the measured absorption coefficient data using a spline fit.
        """
        # Load the random incidence absorption coefficient data included in the GRAS database:
        csvData = pandas.read_csv(self.database + '_csv' + os.sep + 'mat_scene09_plaster.csv', header=None).T
        fMeas = csvData[0]  # Third-octave band center frequencies
        aMeas = csvData[1]  # Third-octave band center absorption coefficients
        sMeas = csvData[2]  # Third-octave band center scattering coefficients

        # Convert to purely real admittance assuming material follows '55 degree rule':
        YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

        # Interpolate to specific frequency list using a spline fit:
        Yf = CubicSpline(fMeas, YsMeas, bc_type='natural')
        Sf = CubicSpline(fMeas, sMeas, bc_type='natural')
        YsInterp = Yf(self.freq)
        SsInterp = Sf(self.freq)

        self.z = 1 / YsInterp
        self.s = SsInterp

    def mdf(self):
        """
        This is a model of the MDF material defined in Scene 9 of the GRAS database. It is a purely real (resistive)
        admittance found from the measured absorption coefficient data using a spline fit.
        """
        # Load the random incidence absorption coefficient data included in the GRAS database:
        csvData = pandas.read_csv(self.database + '_csv' + os.sep + 'mat_MDF25mmA_plane_00deg.csv', header=None).T
        fMeas = csvData[0]  # Third-octave band center frequencies
        aMeas = csvData[1]  # Third-octave band center absorption coefficients
        sMeas = csvData[2]  # Third-octave band center scattering coefficients

        # Convert to purely real admittance assuming material follows '55 degree rule':
        YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

        # Interpolate to specific frequency list using a spline fit:
        Yf = CubicSpline(fMeas, YsMeas, bc_type='natural')
        Sf = CubicSpline(fMeas, sMeas, bc_type='natural')
        YsInterp = Yf(self.freq)
        SsInterp = Sf(self.freq)

        self.z = 1 / YsInterp
        self.s = SsInterp

    def door(self, SampleRate=44100, CrossoverFrequency=250, rho_m=375, d=0.043, A=2.2*0.97, fRes=95, smooth=False):
        """
        This is a model of the door material defined in Scene 9 of the GRAS
        database. It is entirely fabricated from other datasets, since no data
        was given for this component in the GRAS database. It attempts to define
        realistic boundary conditions in a case where insufficient data exists,
        and is included in this work to illustrate the sort of compromises that
        are often necessary, rather than to propose a specific model for these
        materials. The reader is asked to consider it with these caveats in mind.

        It comprises two approaches:

        1) A purely resistive fit to octave-band summed absorption and
        transmission coefficient data. Both absorption and transmission
        coefficients were used since the former did not rise at low frequencies,
        indicating that the data in the dataset use was most likely measured for
        doors on the floor of a reverberation room, hence transmission would be
        zero. From the perspective of this application, tranmission is another
        mechanism by which energy is lost and should be included in absorption,
        hence the coefficients are summed.

        2) A reactive Mass-Spring-Damper model of the assumed fundamental
        resonance of the door panel. This was included since such effects are
        well known to be reactive, and this affects room modal frequencies. The
        Mass value was chosen to be consistent with the assumed material.
        Stiffness and Damping values were tuned to the desired absorption peak
        frequency and bandwidth. This did not however produce sufficient
        absorption to join with the trend in 1, so an additional amount of purely
        resistive absorption was also added.

        These are combined using the non-linear crossover of Aretz el al.

        Parameters
        ----------
        SampleRate : int, optional
            Sampling rate [Hz]
        CrossoverFrequency : int, optional
            Crossover frequency between the models [Hz]
        rho_m : int or float, optional
            Assumed bulk density [kg/m^3]
        d : float, optional
            Assumed thickness [m]
        A : float, optional
            Area [m^2]
        fRes : int, optional
            Assumed fundamental panel resonance frequency [Hz]
        smooth : bool, optional
            Boolean to choose whether apply smoothing to the curve or not.

        Returns
        -------
        Nothing.
        """
        # Model 1: purely resistive fit to octave-band absorption data:

        # Measured data:
        fMeas = [125, 250, 500, 1000, 2000, 4000, ]  # Octave band centre frequencies (Hz)
        aMeas = np.asarray([0.14, 0.10, 0.06, 0.08, 0.1, 0.1, ]) + \
                np.asarray([0.07, 0.01, 0.02, 0.03, 0.01, 0.01, ])  # Absortion and Transmission coefficients

        # Convert to purely real admittance assuming material follows '55 degree rule':
        YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

        # Interpolate to specific frequency list using a spline fit:
        Yf = CubicSpline(fMeas, YsMeas, bc_type='natural')
        Ys1 = Yf(self.freq)

        # Model 2: reactive Mass-Spring-Damper fit to assumed fundamental panel resonance:

        M = rho_m * d * A  # Mass term
        K = M * (2 * np.pi * fRes) ** 2  # Stiffness term  - adjusted to match assumed fRes
        R = 12000  # Resistance term - adjusted to match measured coefficients
        zS = (-1j * 2 * np.pi * self.freq) * M + R + K / (-1j * 2 * np.pi * self.freq)  # Surface impedance
        Ys2 = self.rho0 * self.c0 / zS  # Specific admittance

        # Additional resistive component:
        aExtra = np.mean(aMeas[2::])
        YsExtra = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aExtra)) / (1 + np.sqrt(1 - aExtra))
        Ys2 = Ys2 + YsExtra

        # Define Butterworth filters.
        # Note these are applied twice to make Linkwitz-Riley:
        B_HP, A_HP = butter(8, CrossoverFrequency * 2 / SampleRate, 'high')
        B_LP, A_LP = butter(8, CrossoverFrequency * 2 / SampleRate, 'low')

        # Non-linear crossover method of Aretz et al:
        Ys = np.abs(Ys2 * np.conj(freqz(B_LP, A_LP, self.freq, fs=SampleRate)[1]) ** 2) + \
             np.abs(Ys1 * np.conj(freqz(B_HP, A_HP, self.freq, fs=SampleRate)[1]) ** 2)  # Add the magnitudes only

        Ys = Ys * np.exp(1j * np.angle(Ys2))  # Multiply the phase from MSD model back in

        if smooth:
            Ys_real = savgol_filter(np.real(Ys), 31, 3)
            Ys_imag = savgol_filter(np.imag(Ys), 31, 3)
            Ys = Ys_real + 1j * Ys_imag

        self.z = 1 / Ys

    def window(self, SampleRate=44100, CrossoverFrequency=200, rho_m=2500, d=0.0067, A=5.33, fRes=6.66, smooth=False):
        """
        This is a model of the windows material defined in Scene 9 of the GRAS
        database. It combines two approaches:

        1) A purely resistive fit to teh third-octave band absorption coefficient
        data provided with the GRAS dataset.

        2) A reactive Mass-Spring-Damper model of the assumed fundamental
        resonance of the window panels. This was included since such effects are
        well known be reactive, and this affects room modal frequencies. It
        was also deemed neceessary since the fundamental resonance of the panels
        appeared to be lower than the bandwidth the measured dataset extended to
        (absorption rose quite sharply at the lowest frequencies). The
        Mass value was chosen to be consistent with the assumed material.
        Stiffness and Damping values were tuned to the desired absorption peak
        frequency and bandwidth. This did not however produce sufficient
        absorption to join with the trend in 1, so an additional amount of purely
        resistive absorption was also added.

        These are combined using the non-linear crossover of Aretz el al.

        Note that this script attempts to define realistic boundary conditions in
        a case where insufficient data exists, and is included in this work to
        illustrate the sort of compromises that are often necessary, rather than
        to propose a specific model for these materials. The reader is asked to
        consider it with these caveats in mind.

        Input data:
         - SampleRate = 44100 # Sample rate [Hz]
         - rho_m = 2500  # Assumed bulk density [kg/m^3]
         - d = 0.0067  # Assumed glazing thickness [m]
         - A = 5.33  # Area of each of the three panels [m^2]
         - fRes = 6.66  # Assumed fundamental panel resonance frequency [Hz]
        """
        # Model 1: purely resistive fit to provided third-octave-band absorption data:

        # Load the random incidence absorption coefficient data included in the GRAS database:
        csvData = pandas.read_csv(self.database + '_csv' + os.sep + 'mat_scene09_windows.csv', header=None).T
        fMeas = csvData[0]  # Third-octave band center frequencies
        aMeas = csvData[1]  # Third-octave band center absorption coefficients
        sMeas = csvData[2]  # Third-octave band center scattering coefficients

        # Convert to purely real admittance assuming material follows '55 degree rule':
        YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

        # Interpolate to specific frequency list using a spline fit:
        Yf = CubicSpline(fMeas, YsMeas, bc_type='natural')
        Sf = CubicSpline(fMeas, sMeas, bc_type='natural')
        Ys1 = Yf(self.freq)
        SsInterp = Sf(self.freq)

        self.s = SsInterp

        # Model 2: reactive Mass-Spring-Damper fit to assumed fundamental panel resonance:

        M = rho_m * d * A  # Mass term
        K = M * (2 * np.pi * fRes) ** 2  # Stiffness term  - adjusted to match assumed fRes
        R = 6000  # Resistance term - adjusted to match measured coefficients
        zS = (-1j * 2 * np.pi * self.freq) * M + R + K / (-1j * 2 * np.pi * self.freq)  # Surface impedance
        Ys2 = self.rho0 * self.c0 / zS  # Specific admittance

        # Additional resistive component:
        aExtra = aMeas[8]
        YsExtra = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aExtra)) / (1 + np.sqrt(1 - aExtra))
        Ys2 = Ys2 + YsExtra

        # Define Butterworth filters.
        # Note these are applied twice to make Linkwitz-Riley:
        B_HP, A_HP = butter(8, CrossoverFrequency * 2 / SampleRate, 'high')
        B_LP, A_LP = butter(8, CrossoverFrequency * 2 / SampleRate, 'low')

        # Non-linear crossover method of Aretz et al:
        Ys = np.abs(Ys2 * np.conj(freqz(B_LP, A_LP, self.freq, fs=SampleRate)[1]) ** 2) + \
             np.abs(Ys1 * np.conj(freqz(B_HP, A_HP, self.freq, fs=SampleRate)[1]) ** 2)  # Add the magnitudes only

        Ys = Ys * np.exp(1j * np.angle(Ys2))  # Multiply the phase from MSD model back in

        if smooth:
            Ys_real = savgol_filter(np.real(Ys), 31, 3)
            Ys_imag = savgol_filter(np.imag(Ys), 31, 3)
            Ys = Ys_real + 1j * Ys_imag

        self.z = 1 / Ys

    def filter_alpha(self, nthOct=1, plot=True, returnValues=False, show=False, figsize=(15, 5)):
        """
        Filters the absorption coefficient into nthOct bands.

        Parameters
        ----------
        nthOct : int, optional
            Fractional octave bands that the absorption will be filtered to.
        plot : bool, optional
            Boolean to display plot with filtered absorption.
        returnValues : bool, optional
            Boolean to return the bands and filetered values.
        show : bool, optional
            Boolean to display the filtered values in a table.
        figsize : tuple, optional
            Tuple containing the width and height of the figure.

        Returns
        -------
        bands : ndarray
            An array containing the center frequencies of the available bands.
        result : ndarray
            An array containing the filtered absorption coefficient in the available bands.
        """
        bands, result = pytta.utils.filter_values(self.freq, self.alpha, nthOct=nthOct)

        # Plot
        if plot:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax1.semilogx(self.freq, self.alpha, label='Narrowband')
            ax2 = ax1.twiny()
            ax2.set_xscale('log')
            ax1.semilogx(bands, result, 'o-', label=f'1/{nthOct} octave band')
            ax2.set_xticks([freq for freq in bands.tolist()])
            ax2.set_xticklabels([f'{freq:0.1f}' for freq in bands.tolist()])
            ax2.set_xlim(ax1.get_xlim())
            ax1.set_ylabel('Absorption Coefficient [-]')
            ax1.set_xlabel('Narrowband Frequency [Hz]')
            ax2.set_xlabel(f'1/{nthOct} Octave Bands [Hz]')
            ax1.set_ylim([-0.1, 1.1])
            ax1.legend(loc='best')
            ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax1.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax1.tick_params(which='minor', length=5, rotation=-90,
                            axis='x')  # Set major and minor ticks to same length
            ax1.tick_params(which='major', length=5, rotation=-90,
                            axis='x')  # Set major and minor ticks to same length
            ax2.tick_params(which='major', length=5, rotation=-90,
                            axis='x')  # Set major and minor ticks to same length
            ax1.minorticks_on()  # Set major and minor ticks to same length
            ax2.minorticks_off()
            ax1.grid('minor')
            plt.show()

        if show:
            pandas.set_option("display.precision", 2)
            freq_bands = []
            absorption = []
            absorption_percentual = []
            #             for key, value in available_data.items():
            for i in range(len(bands)):
                freq_bands.append(float(f'{bands[i]:0.2f}'))
                absorption.append(float(f'{result[i]:0.2f}'))
                absorption_percentual.append(float(f'{result[i] * 100:0.0f}'))
            data = {'Bands [Hz]': freq_bands, 'Absorption [-]': absorption, 'Absorption [%]': absorption_percentual}
            df = pandas.DataFrame(data=data).set_index('Bands [Hz]').T
            df = df.style.set_caption(f'1/{nthOct} Octave Absorption Data')

            try:
                from IPython.display import display
                display(df)
            except:
                print('IPython.diplay unavailable.')

        if returnValues:
            return bands, result

    def save2sheet(self, filename='MaterialModel', timestamp=True, ext='.xlsx', chart_styles=[35, 36], nthOct=3):

        timestr = time.strftime("%Y%m%d-%H%M_")
        if self.project_folder is None:
            full_path = outputs + os.sep + filename + ext
            if timestamp is True:
                full_path = outputs + os.sep + timestr + filename + ext
        else:
            folderCheck = os.path.exists(self.project_folder + os.sep + 'Treatments')
            if folderCheck is False:
                os.mkdir(self.project_folder + os.sep + 'Treatments')
            full_path = self.project_folder + os.sep + 'Treatments' + os.sep + filename + ext
            if timestamp is True:
                full_path = self.project_folder + os.sep + 'Treatments' + os.sep + timestr + filename + ext

        workbook = xlsxwriter.Workbook(full_path)
        worksheet = workbook.add_worksheet()

        # Setting formats
        bold = workbook.add_format({'bold': True, 'font_color': 'black', 'align': 'center', 'border': 2})
        regular = workbook.add_format({'bold': False, 'font_color': 'black', 'align': 'center', 'border': 1})

        # Adding frequency related data
        worksheet.write(0, 0, 'Frequency', bold)
        worksheet.write(0, 1, 'Real Z', bold)
        worksheet.write(0, 2, 'Img Z', bold)
        worksheet.write(0, 3, 'Absorption', bold)
        for i in range(len(self.freq)):
            worksheet.write(1 + i, 0, self.freq[i], regular)
            worksheet.write(1 + i, 1, np.real(self.z[i]), regular)
            worksheet.write(1 + i, 2, np.imag(self.z[i]), regular)
            worksheet.write(1 + i, 3, self.alpha[i], regular)

        # Absorption coefficient plot
        chart_abs = workbook.add_chart({'type': 'line'})
        chart_abs.add_series({'name': ['Sheet1', 0, 3],
                              'categories': ['Sheet1', 1, 0, len(self.freq) + 1, 0],
                              'values': ['Sheet1', 1, 3, len(self.freq) + 1, 3], })
        chart_abs.set_title({'name': 'Absorption Coefficient'})
        chart_abs.set_x_axis({'name': 'Frequency [Hz]'})
        chart_abs.set_y_axis({'name': 'Alpha [-]'})
        chart_abs.set_style(chart_styles[0])
        worksheet.insert_chart('G1', chart_abs, {'x_offset': 0, 'y_offset': 0, 'x_scale': 1.334, 'y_scale': 1.11})

        # Impedance plot
        chart_z = workbook.add_chart({'type': 'line'})
        chart_z.add_series({'name': ['Sheet1', 0, 1],
                            'categories': ['Sheet1', 1, 0, len(self.freq) + 1, 0],
                            'values': ['Sheet1', 1, 1, len(self.freq) + 1, 1], })
        chart_z.add_series({'name': ['Sheet1', 0, 2],
                            'categories': ['Sheet1', 1, 0, len(self.freq) + 1, 0],
                            'values': ['Sheet1', 1, 2, len(self.freq) + 1, 2], })
        chart_z.set_title({'name': 'Normalized Surface Impedance'})
        chart_z.set_x_axis({'name': 'Frequency [Hz]'})
        chart_z.set_y_axis({'name': 'Z [Pa*s/m]'})
        chart_z.set_style(chart_styles[1])
        worksheet.insert_chart('G17', chart_z, {'x_offset': 0, 'y_offset': 0, 'x_scale': 1.334, 'y_scale': 1.11})

        # Adding nthOct band absorption coeffiecients
        line = 0
        idx = 4
        worksheet.merge_range(line, idx, line, idx + 1, f'1/{nthOct} octave band absorption coefficients', bold)
        line += 1
        worksheet.write(line, idx, 'Frequency Band [Hz]', bold)
        worksheet.write(line, idx + 1, 'Absorption Coeffiecient [-]', bold)
        line += 1
        xOct, yOct = self.filter_alpha(nthOct=nthOct, plot=False, returnValues=True)
        for x, y in zip(xOct, yOct):
            worksheet.write(line, idx, x, regular)
            worksheet.write(line, idx + 1, y, regular)
            line += 1

        # Setting column widths
        worksheet.set_column('A:D', 12)
        worksheet.set_column('E:F', 28)

        workbook.close()

        print(f'Sheet saved to ', full_path)

    def plot(self, figsize=(15, 5), plots=['z', 'y', 'alpha'], saveFig=False, filename='TMM', timestamp=True,
             ext='.png'):
        """
        Plots impedance, admittance and absorption curves.

        Parameters
        ----------
        figsize : tuple, optional
            Tuple containing the width and height of the figure.
        plots : list of strings, optional
            Desired curves to be plotted. 'z' for impedance, 'y' for admittance and 'alpha' for absorption.
        saveFig : bool, optional
            Option to save the plot as an image.
        filename : strint, optional
            Name to be added in the filename:
        timestamp : bool, optional
            Boolean to add timestamping to the filename.
        ext : string, optional
            Desired file extension.
        max_mode : None, int or string
            Variable to set a maximum limit to peak detection in the absorption coefficient.
            'all' for no limit, None for  no detection or int for maximum detection frequency.

        Returns
        -------
        Nothing.
        """

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, len(plots))

        i = 0
        if 'z' in plots or 'Z' in plots:
            ax_z = plt.subplot(gs[0, i])
            ax_z.set_title(r'Impedance ($Z$)')
            ax_z.set_xlabel('Frequency [Hz]')
            ax_z.set_ylabel('Normalized Surface Impedance [Z/Z0]')
            ax_z.semilogx(self.freq, np.real(self.z), linewidth=2, label='Real')
            ax_z.semilogx(self.freq, np.imag(self.z), linewidth=2, label='Real')
            ax_z.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_z.axhline(y=0, color='k', linewidth=0.5)
            ax_z.axhline(y=1, linestyle='--', color='gray')
            ax_z.legend(loc='best')
            ax_z.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_z.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_z.tick_params(which='minor', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_z.tick_params(which='major', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_z.minorticks_on()  # Set major and minor ticks to same length
            ax_z.grid('minor')
            i += 1

        if 'y' in plots or 'Y' in plots:
            ax_y = plt.subplot(gs[0, i])
            ax_y.set_title(r'Admittance ($Y$)')
            ax_y.set_xlabel('Frequency [Hz]')
            ax_y.set_ylabel('Normalized Surface Admittance [Z0/Z]')
            ax_y.semilogx(self.freq, np.real(self.y), linewidth=2, label='Real')
            ax_y.semilogx(self.freq, np.imag(self.y), linewidth=2, label='Imag')
            ax_y.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_y.legend(loc='best')
            ax_y.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_y.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_y.tick_params(which='minor', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_y.tick_params(which='major', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_y.minorticks_on()  # Set major and minor ticks to same length
            ax_y.grid('minor')
            i += 1

        if 'alpha' in plots or 'abs' in plots or 'scattering' in plots or 'scat' in plots or 's' in plots:
            ax_a = plt.subplot(gs[0, i])
            ax_a.set_xlabel('Frequency [Hz]')
            ax_a.set_ylabel('Coefficient [-]')
            alpha = False
            scat = False
            if 'alpha' in plots or 'abs' in plots:
                ax_a.semilogx(self.freq, self.alpha, linewidth=2, label='Absorption')
                alpha = True
            if self.s is not None:
                if 'scattering' in plots or 'scat' in plots or 's' in plots:
                    ax_a.semilogx(self.freq, self.s, linewidth=2, label='Scattering')
                    scat = True
            if alpha:
                ax_a.set_title(r'Absorption Coefficient ($\alpha$)')
            if scat:
                ax_a.set_title(r'Scattering Coefficient')
            if alpha and scat:
                ax_a.set_title(r'Absorption Coefficient & Scattering Coefficients')
            ax_a.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_a.set_ylim([-0.1, 1.1])
            ax_a.legend(loc='best')
            ax_a.axhline(y=0, color='k', linewidth=0.5)
            ax_a.axhline(y=1, linestyle='--', color='gray')
            ax_a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_a.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_a.tick_params(which='minor', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_a.tick_params(which='major', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_a.minorticks_on()  # Set major and minor ticks to same length
            ax_a.grid('minor')
            i += 1

        gs.tight_layout(fig, pad=4, w_pad=1, h_pad=1)

        if saveFig:
            timestr = time.strftime("%Y%m%d-%H%M_")
            if self.project_folder is None:
                full_path = outputs + os.sep + filename + ext
                if timestamp is True:
                    full_path = outputs + os.sep + timestr + filename + ext
            else:
                folderCheck = os.path.exists(self.project_folder + os.sep + 'Treatments')
                if folderCheck is False:
                    os.mkdir(self.project_folder + os.sep + 'Treatments')
                full_path = self.project_folder + os.sep + 'Treatments' + os.sep + filename + ext
                if timestamp is True:
                    full_path = self.project_folder + os.sep + 'Treatments' + os.sep + timestr + filename + ext

            plt.savefig(full_path, dpi=100)
            print('Image saved to ', full_path)
        plt.show()


if __name__ == '__main__':

    from material_model import MaterialModel

    # Define the frequency range, resolution and project parameters
    mm = MaterialModel(fmin=20, fmax=5000, df=1, c0=343, rho0=1.21)

    # Choose the material
    mm.door()
    mm.plot(figsize=(7, 5), plots=['alpha'], saveFig=True, filename='example_door', timestamp=False)
    mm.save2sheet(timestamp=False, filename='example_door', nthOct=1)
    bands, filtered_alpha = mm.filter_alpha(figsize=(7, 5),
                                            plot='available',
                                            show=True,
                                            nthOct=1,
                                            returnValues=True)