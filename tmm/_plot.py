""""
Data visualization module.

This module receives organized raw data and plot it through its functions.

For further information check the function specific documentation.
"""
import os
import numpy as np
import time
from matplotlib import ticker, gridspec, style
from matplotlib import pyplot as plt
from tmm import _utils as utils
style.use("seaborn-colorblind")


def save_matplotlib_fig(fig, filename, project_folder, timestamp=False, subfolder="", ext=".png", **kwargs):
    """
    Save Matplotlib figure to a static image file.

    Parameters
    ----------
    filename : str
        String that will be used as the full or partial file name.
    project_folder : str
        String containing the destination folder in which the image will be saved. If not defined the current active
        folder will be used.
    timestamp : bool, optional
        Boolean to add timestamping to the filename.
    subfolder : str, optional
        Subfolder in which the image will be saved.
    ext : string, optional
        Desired file extension.
    kwargs : keyword arguments, optional
        See matplotlib.pyplot.savefig.
    """
    timestr = time.strftime("%Y%m%d-%H%M_")
    subfolder = os.sep + subfolder if subfolder != "" else subfolder
    folder_check = os.path.exists(project_folder + subfolder)
    if folder_check is False:
        os.mkdir(project_folder + subfolder)
    full_path = project_folder + subfolder + os.sep + filename + ext
    if timestamp is True:
        full_path = project_folder + subfolder + os.sep + timestr + filename + ext

    if "dpi" not in kwargs:
        kwargs["dpi"] = 150
    if "transparent" not in kwargs:
        kwargs["transparent"] = True
    if "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = "tight"

    fig.savefig(full_path, **kwargs)
    print("Image saved to ", full_path)


def acoustic_data(tmms, fig=None, ax=None, gs=None, figsize=(16, 9), plots=None, max_mode=True, show_incidence=True,
                  labels=True, orientation="vertical", base_fontsize=12, xtype="log", legend="inside", save_fig=False,
                  project_folder=None, filename=None,  **kwargs):
    """
    Plots impedance, admittance, absorption and scattering coefficients with Matplotlib.

    Parameters
    ----------
    tmms : list
        List of TMM class objects.
    fig : class
        Matplotlib Figure object.
    ax : class
        Matplotlib Axes object.
    gs : class
        Matplotlib GridSpec object.
    figsize : tuple, optional
        Figure size.
    plots : list, optional
        List of strings with the desired plots - 'z' for impedance, 'y' for admittance, 'alpha' for absorption
        coefficient and 'scat' for scattering coefficient.
    max_mode : bool, optional
        Option to identify first absorption peak if any.
    show_incidence : bool, optional
        Option to display both normal and diffuse incidence absorption curves.
    labels : bool or str, optional
        Option to display the full filename of the TMM object in the label.
    orientation : str, optional
        Stacking orientation of the subplot - 'vertical' or 'horizontal'.
    base_fontsize : int, optional
        Base font size.
    xtype : string, optional
        Frequency axis type - 'linear' or 'log'.
    legend : string, optional
        Legend placement option - 'inside' or 'outside'.
    save_fig : bool, optional
        Option to save figure as static image.
    project_folder : str
        String containing the destination folder in which the image will be saved. If not defined the current active
        folder will be used.
    filename : str
        String that will be used as the full or partial file name.
    kwargs : keyword arguments, optional
        See tmm._plot.save_matplotlib_fig.

    Returns
    -------
    Matplolib Figure, list of axes Axes and GridSpec objects.
    """
    if plots is None:
        plots = ["z", "y", "alpha"]
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = [None for i in range(len(plots))]
    if gs is None:
        if orientation == "vertical":
            gs = gridspec.GridSpec(len(plots), 1)
        elif orientation == "horizontal":
            gs = gridspec.GridSpec(1, len(plots))
        else:
            raise ValueError("Orientation must be either 'vertical' or 'horizontal'.")

    for tmm in tmms:
        if labels == "full" and show_incidence:
            label_name = "" if tmm._filename is None else tmm.filename + " | "
        elif labels == "full":
            label_name = "" if tmm._filename is None else tmm.filename
        else:
            label_name = ""
        i = 0
        if "z" in plots or "Z" in plots:
            if ax[i] is None:
                if orientation == "vertical":
                    ax[i] = plt.subplot(gs[i, 0])
                elif orientation == "horizontal":
                    ax[i] = plt.subplot(gs[0, i])
            ax[i].set_title(r"Normalized Surface Impedance ($Z$)", fontsize=base_fontsize)
            ax[i].set_ylabel(r"$Z$ [Pa·s/m³]", fontsize=base_fontsize - 2)
            ax[i].plot(tmm.freq, np.real(tmm.z_norm), linewidth=2, label=label_name + "Real", c=tmm.color,
                           linestyle='-')
            ax[i].plot(tmm.freq, np.imag(tmm.z_norm), linewidth=2, label=label_name + "Imag.", c=tmm.color,
                           linestyle='--')
            i += 1

        if "y" in plots or "Y" in plots:
            if ax[i] is None:
                if orientation == "vertical":
                    ax[i] = plt.subplot(gs[i, 0])
                elif orientation == "horizontal":
                    ax[i] = plt.subplot(gs[0, i])
            ax[i].set_title(r"Normalized Surface Admittance ($Y$)", fontsize=base_fontsize)
            ax[i].set_ylabel(r"$Y$ [m³/Pa·s]", fontsize=base_fontsize - 1)
            ax[i].plot(tmm.freq, np.real(tmm.y_norm), linewidth=2, label=label_name + "Real", c=tmm.color,
                           linestyle='-')
            ax[i].plot(tmm.freq, np.imag(tmm.y_norm), linewidth=2, label=label_name + "Imag.", c=tmm.color,
                           linestyle='--')
            i += 1

        if "alpha" in plots or "abs" in plots:
            if ax[i] is None:
                if orientation == "vertical":
                    ax[i] = plt.subplot(gs[i, 0])
                elif orientation == "horizontal":
                    ax[i] = plt.subplot(gs[0, i])
            ax[i].set_title(r"Absorption Coefficient ($\alpha$)", fontsize=base_fontsize, loc="left")
            ax[i].set_ylabel(r"$\alpha$ [-]", fontsize=base_fontsize - 1)
            if show_incidence:
                if tmm.incidence == "diffuse" and "material_model" not in tmm.filename:
                    ax[i].plot(tmm.freq, tmm.alpha, linewidth=2, c=tmm.color,
                                   label=label_name + "Diffuse Incidence " +
                                         f"({min(tmm.incidence_angle):0.0f}° - {max(tmm.incidence_angle):0.0f}°)")
                    if np.round(tmm.incidence_angle[0]) == 0:
                        ax[i].plot(tmm.freq, tmm.alpha_angle(), linewidth=2, c=tmm.color,
                                       label=label_name + "Normal Incidence", linestyle="--")
                    else:
                        ax[i].plot(tmm.freq, tmm.alpha_angle(), linewidth=2, c=tmm.color,
                                       label=label_name + f"Incidence at {tmm.incidence_angle[0]:0.0f}°",
                                       linestyle="--")
                else:
                    ax[i].plot(tmm.freq, tmm.alpha, linewidth=2, label=label_name,
                                   c=tmm.color)
            else:
                ax[i].plot(tmm.freq, tmm.alpha, linewidth=2, label=label_name, c=tmm.color)
            if max_mode and tmm.first_peak[0] != max(tmm.freq):
                ax[i].axvline(x=tmm.first_peak[0],
                              label=f"Resonance at {tmm.first_peak[0]} Hz",
                              linestyle=":",
                              color="green")
            ax[i].set_ylim([-0.1, 1.1])
            ax[i].yaxis.set_ticks(np.arange(0, 1.01, 0.1))
            ax[i].set_yticklabels(np.arange(0, 1.01, 0.1))
            ax[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            i += 1

        if "scat" in plots or "scattering" in plots:
            if ax[i] is None:
                if orientation == "vertical":
                    ax[i] = plt.subplot(gs[i, 0])
                elif orientation == "horizontal":
                    ax[i] = plt.subplot(gs[0, i])
            ax[i].set_title(r"Scattering Coefficient ($s$)", fontsize=base_fontsize)
            ax[i].set_ylabel(r"$s$ [-]", fontsize=base_fontsize - 1)
            ax[i].plot(tmm.freq, tmm.scat, linewidth=2, label=label_name, c=tmm.color)
            ax[i].set_ylim([-0.1, 1.1])
            ax[i].yaxis.set_ticks(np.arange(0, 1.01, 0.1))
            ax[i].set_yticklabels(np.arange(0, 1.01, 0.1))
            ax[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            i += 1

        for i in range(len(ax)):
            ax[i].set_xlabel("Frequency [Hz]", fontsize=base_fontsize - 1)
            ax[i].set_xscale(xtype)
            ax[i].set_xlim([(np.min(tmm.freq)), (np.max(tmm.freq))])
            ax[i].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax[i].get_xaxis().set_minor_formatter(ticker.ScalarFormatter())
            ax[i].tick_params(which="minor", length=5, rotation=-90, axis="x")
            ax[i].tick_params(which="major", length=5, rotation=-90, axis="x")
            ax[i].tick_params(axis="both", which="both", labelsize=base_fontsize - 2)
            ax[i].minorticks_on()
            ax[i].grid("minor")

            if labels == "full" and orientation == "vertical":
                if legend == "outside":
                    ax[i].legend(bbox_to_anchor=(1.04, 0), loc="lower left", fontsize=base_fontsize - 2)
                else:
                    ax[i].legend(loc="best", fontsize=base_fontsize - 2)
            elif labels is True:
                ax[i].legend(loc="best", fontsize=base_fontsize - 2)

    gs.tight_layout(fig, pad=4, w_pad=1, h_pad=1)

    if save_fig:
        filename = filename if filename is not None else "acoustic_data"
        project_folder = project_folder if project_folder is not None else os.getcwd()
        if "subfolder" not in kwargs:
            kwargs["subfolder"] = "Treatments"
        save_matplotlib_fig(fig, filename, project_folder, **kwargs)

    return fig, ax, gs


def oct_filter(narrowbad_freq, narrowband_value, freq_bands, filtered_value, n_oct, y_label, x_scale="log",
               figsize=(16, 9), save_fig=False, filename=None, project_folder=None, base_fontsize=12, **kwargs):
    """
    Plot the narrowband data and the octave filtered data together.

    Parameters
    ----------
    narrowbad_freq : array
        1D array of narrowband frequency values.
    narrowband_value : array
        1D array of narrowband amplitude values.
    freq_bands : array
        1D array of frequency bands.
    filtered_value : array
        1D array of octave filtered values.
    n_oct : int
        Number of fractional octaves.
    y_label : str
        Y axis label.
    x_scale : str
        X axis scale type.
    figsize : tuple, optional
        Figure size.
    save_fig : bool, optional
        Option to save figure as static image.
    filename : str
        String that will be used as the full or partial file name.
    project_folder : str
        String containing the destination folder in which the image will be saved. If not defined the current active
        folder will be used.
    base_fontsize : int, optional
        Base font size.
    kwargs : keyword arguments, optional
        See tmm._plot.save_matplotlib_fig.

    Returns
    -------
    Matplolib Figure and list of Axes objects.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.semilogx(narrowbad_freq, narrowband_value, label="Narrowband")
    ax1.semilogx(freq_bands, filtered_value, "o-", label=f"1/{n_oct} octave band")
    ax1.set_ylabel(y_label, fontsize=base_fontsize - 1)
    ax1.set_xlabel("Narrowband Frequency [Hz]", fontsize=base_fontsize - 1)
    ax1.set_ylim([-0.1, 1.1])
    ax1.legend(loc="best", fontsize=base_fontsize - 2)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())
    ax1.tick_params(which="minor", length=5, rotation=-90, axis="x")
    ax1.tick_params(which="major", length=5, rotation=-90, axis="x")
    ax1.tick_params(axis='both', labelsize=base_fontsize - 2)
    ax1.yaxis.set_ticks(np.arange(0, 1.01, 0.1))
    ax1.set_yticklabels(np.arange(0, 1.01, 0.1))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax1.minorticks_on()
    ax1.grid("minor")

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xscale(x_scale)
    ax2.set_xlabel(f"1/{n_oct} Octave Bands [Hz]", fontsize=base_fontsize)
    ax2.set_xticks([freq for freq in freq_bands.tolist()])
    bands_ticklabels = [f"{freq:0.1f}" for freq in freq_bands.tolist()]
    for b in range(len(bands_ticklabels)):
        bands_ticklabels[b] = int(freq_bands.tolist()[b]) if float(freq_bands.tolist()[b]).is_integer() is True \
            else float(bands_ticklabels[b])
    ax2.set_xticklabels(bands_ticklabels)
    ax2.tick_params(which="major", length=5, rotation=-90, axis="x")
    ax2.tick_params(axis='both', labelsize=base_fontsize - 2)
    ax2.minorticks_off()

    if save_fig:
        filename = filename if filename is not None else "oct_filter"
        project_folder = project_folder if project_folder is not None else os.getcwd()
        if "subfolder" not in kwargs:
            kwargs["subfolder"] = "Treatments"
        save_matplotlib_fig(fig, filename, project_folder, **kwargs)

    return fig, [ax1, ax2]

