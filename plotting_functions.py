import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import scipy.stats as st
import os
import matplotlib.patches as mpatches
import seaborn as sns
from helper_functions import point_estimate


def plot_kde_combined(input_data, eye_data, include_progressive_saccades=True, safe_plot=True):
    # reducing data to only respective events of interest
    inputs = input_data[input_data["start_input"] == 1.0]
    saccades = eye_data[eye_data["saccadeOnset"] == 1.0]

    # define arrays of time tags for respective data
    input_data_array = np.asarray(inputs.time_played)
    eye_data_array = np.asarray(saccades.time_tag)

    # define point dataframes
    input_data_points = {'x': input_data_array, 'y': [0] * len(input_data_array)}
    input_data_points = pd.DataFrame(data=input_data_points)
    # dataframe for eye_data points is generated down below...

    # compute hpdi (I went for the smallest interval which contains 25% of the mass)
    input_data_hpdi_bounds = az.hdi(input_data_array, 0.25)
    eye_data_hpdi_bounds = az.hdi(eye_data_array, 0.25)

    # plot boundaries:
    lbound = 0
    ubound = input_data.iloc[-2].time_played  # second last row because last row is written to df AFTER SoC response given which may take time

    # instatiate KDEs
    kde_init = np.linspace(lbound, ubound, 100)

    input_data_kde = st.gaussian_kde(input_data_array)
    eye_data_kde = st.gaussian_kde(eye_data_array)

    # Grid
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(
        f"Densities of events for trial {inputs.iloc[0].trial} (drift enabled = {inputs.iloc[0].drift_enabled}, input noise = {inputs.iloc[0].input_noise_magnitude}, crashed = {input_data.iloc[-1].collision})",
        fontdict={"fontweight": "bold"})

    # axis labels
    ax.set_xlabel("Time played")
    ax.set_ylabel("Density")

    ax.set_xlim([lbound, ubound])

    xaxis = np.linspace(lbound, ubound, 10)
    ax.set_xticks(xaxis)

    # Plotting
    colors = ["crimson", "limegreen", "royalblue"]

    ax.plot(kde_init, input_data_kde(kde_init), color=colors[0], label='input data')
    # ax.fill_between(kde_init, input_data_kde(kde_init), step="mid", alpha=0.3, color=colors[0])
    ax.scatter(input_data_points.x, input_data_points.y, marker=".", color=colors[0])

    ax.plot(kde_init, eye_data_kde(kde_init), color=colors[1], label='eye movement data')
    # ax.fill_between(kde_init, eye_data_kde(kde_init), step="mid", alpha=0.3, color=colors[1])

    # define dataframe for points of eye_data ( now because we can retrieve y-axis limits at this point)
    y_max = ax.get_ylim()[1]  # 0: bottom; 1: top
    eye_data_points = {'x': eye_data_array,
                       'y': [y_max / 99] * len(eye_data_array)}  # y_max/99 to plot these a bit above input data points
    eye_data_points = pd.DataFrame(data=eye_data_points)
    ax.scatter(eye_data_points.x, eye_data_points.y, marker=".", color=colors[1])

    # HPDIs:
    point_estimate_input_data = point_estimate(inputs.time_played)
    # ax.axvspan(point_estimate_input_data[2], point_estimate_input_data[3], alpha=0.3, color=colors[0])
    plt.vlines(point_estimate_input_data[0], ymin=0, ymax=point_estimate_input_data[1], color=colors[0])

    point_estimate_eye_data = point_estimate(saccades.time_tag)
    # ax.axvspan(point_estimate_eye_data[2], point_estimate_eye_data[3], alpha=0.3, color=colors[1])
    plt.vlines(point_estimate_eye_data[0], ymin=0, ymax=point_estimate_eye_data[1], color=colors[1])

    # progressive saccades?
    while include_progressive_saccades:
        progressive_saccades = saccades.loc[saccades["saccade_direction_y"] <= 0]
        progressive_eye_data_array = np.asarray(progressive_saccades.time_tag)

        # early exit out of while loop
        if len(progressive_eye_data_array) <= 1:
            include_progressive_saccades = False
            break

        progressive_eye_data_hpdi_bounds = az.hdi(progressive_eye_data_array, 0.25)
        progressive_eye_data_kde = st.gaussian_kde(progressive_eye_data_array)
        ax.plot(kde_init, progressive_eye_data_kde(kde_init), color=colors[2], label='progressive eye movement data')

        progressive_eye_data_points = {'x': progressive_eye_data_array,
                                       'y': [y_max / 99] * len(progressive_eye_data_array)}
        progressive_eye_data_points = pd.DataFrame(data=progressive_eye_data_points)
        ax.scatter(progressive_eye_data_points.x, progressive_eye_data_points.y, marker=".", color=colors[2])

        point_estimate_progressive_eye_data = point_estimate(progressive_saccades.time_tag)
        plt.vlines(point_estimate_progressive_eye_data[0], ymin=0, ymax=point_estimate_progressive_eye_data[1],
                   color=colors[2])

        # get out of the while loop
        include_progressive_saccades = False

    ax.legend()

    if safe_plot:
        plt.savefig(
            f"{os.getcwd()}/plots/kde_plots_event_distribution/Event_densities_trial {inputs.iloc[0].trial}", dpi=300)
        plt.close()


def plot_fixation_location_kde(eye_data_none, eye_data_weak, eye_data_strong, level=1, drift_enabled=False, safe_plot=False,
                               path_to_save_folder=f"{os.getcwd()}/plots/kde_plots_fixation_locations/"):
    """
    :param eye_data_none: data of onput noise = nan
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """

    # initiate plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 3 subplots
    fig.suptitle(f"fixation locations KDE; level {level}, drift_enabled = {drift_enabled}")

    fig.supxlabel("observation space x")
    fig.supylabel("observation space y")

    # Plotting
    colors = ["crimson", "limegreen", "lightskyblue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["NaN", "weak", "strong"]
    counter = 0

    for eye_data, ax in zip([eye_data_none, eye_data_weak, eye_data_strong], axs.ravel()):
        # axis labels
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")

        ax.set_xlim([-300, 300])
        ax.set_ylim([-600, 600])

        fixations = eye_data[eye_data["Fixation"] == 1.0]

        heatmap = sns.kdeplot(x=fixations.converging_eye_x,
                              y=fixations.converging_eye_y,
                              cmap=color_maps[counter],
                              shade=True,
                              alpha=0.9,
                              bw_adjust=0.4,
                              ax=ax)

        y_coord = point_estimate(fixations.converging_eye_y)[0]
        ax.axhline(y_coord, color=colors[counter])
        x_coord = point_estimate(fixations.converging_eye_x)[0]
        ax.axvline(x_coord, color=colors[counter])

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label=input_noise_magnitude[0]),
               mpatches.Patch(facecolor=plt.cm.Greens(100), label=input_noise_magnitude[1]),
               mpatches.Patch(facecolor=plt.cm.Blues(100), label=input_noise_magnitude[2])]
    fig.legend(handles=handles, loc='center right')

    if safe_plot:
        plt.savefig(f"{path_to_save_folder}fixation_location_kde_level_{level}_drift_enabled_{drift_enabled}", dpi=300)
        plt.close()


def plot_eye_rest_y_over_time(eye_data_none, eye_data_weak, eye_data_strong, input_data, safe_plot=False,
                              path_to_save_folder=f"{os.getcwd()}/plots/eye_resting_position_y/"):
    """
    :param eye_data_none: data of onput noise = nan
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param input_data: from which level features will be extracted as well as drift onset
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """

    # get level features from input data
    ## exemplary input data
    level = input_data.iloc[-1].trial
    drift_enabled = input_data.iloc[-1].drift_enabled

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"eye resting y positions; level = {level}, drift_enabled = {drift_enabled}", fontdict={"fontweight": "bold"})

    # axis labels
    ax.set_xlabel("time_played")
    ax.set_ylabel("observation space y")

    # ax.set_ylim([-70, 70])

    # plt.gca().invert_yaxis()

    # Plotting
    ## initiate colors and labels
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["NaN", "weak", "strong"]

    counter = 0

    if drift_enabled:
        # draw lines where drift tile onset
        drift_tile_onset = input_data[input_data.drift_tile_onset == 1]
        for drift_onset_time_tag in drift_tile_onset.time_played:
            ax.axvline(drift_onset_time_tag, color="pink")

        second_drift_onset = input_data[input_data.second_drift_tile_onset == 1]
        for drift_onset_time_tag in second_drift_onset.time_played:
            ax.axvline(drift_onset_time_tag, color="hotpink")

    # for run in list_of_runs:
    for eye_data in [eye_data_none, eye_data_weak, eye_data_strong]:
        fixations = eye_data[eye_data["Fixation"] == 1]

        # draw on canvas
        ax.plot(fixations.time_tag, fixations.converging_eye_y, color=colors[counter], alpha=0.8)

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label=[input_noise_magnitude[0]]),
               mpatches.Patch(facecolor=plt.cm.Greens(100), label=[input_noise_magnitude[1]]),
               mpatches.Patch(facecolor=plt.cm.Blues(100), label=[input_noise_magnitude[2]])]
    ax.legend(handles=handles)

    if safe_plot:
        plt.savefig(f"{path_to_save_folder}eye_resting_position_y_level_{level}_drift_enabled_{drift_enabled}", dpi=300)
        plt.close()


def plot_saccade_amplitudes(eye_data_none, eye_data_weak, eye_data_strong, level, drift_enabled, safe_plot=False,
                            path_to_save_folder=f"{os.getcwd()}/plots/plots_saccade_amplitude/"):
    """
    :param eye_data_none: data of onput noise = nan
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title(f"saccade amplitudes; level {level}, drift_enabled = {drift_enabled}", fontdict={"fontweight": "bold"})

    # axis labels
    ax.set_xlabel("input noise magnitude")
    ax.set_ylabel("saccade amplitude (in °)")

    ax.set_xlim([-70, 70])
    ax.set_xticks([-50, 0, 50])
    ax.set_xticklabels(["NaN", "weak", "strong"])

    # Plotting
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["NaN", "weak", "strong"]
    offset = [-50, 0, 50]
    n_saccades = [np.nan, np.nan, np.nan]
    n_progressive_saccades = [np.nan, np.nan, np.nan]

    counter = 0

    for eye_data in [eye_data_none, eye_data_weak, eye_data_strong]:
        # all kinds of saccades
        saccades = eye_data[eye_data["saccadeOnset"] >= 1]
        n_saccades[counter] = len(saccades)
        plot_labels = [offset[counter]] * len(saccades)
        saccades["plot_label"] = [offset[counter]] * len(saccades)
        # progressive saccades
        progressive_saccades = saccades.loc[saccades["saccade_direction_y"] <= 0]
        n_progressive_saccades[counter] = len(progressive_saccades)

        # draw on canvas
        ax.scatter(saccades.plot_label, saccades.saccade_amplitude, marker=".", color=colors[counter], alpha=0.3)
        ax.plot(offset[counter], np.mean(saccades.saccade_amplitude), marker=0, markersize=10, color=colors[counter], alpha=1.0)

        ax.scatter(progressive_saccades.plot_label, progressive_saccades.saccade_amplitude, marker=".",
                   color=colors_p[counter], alpha=0.3)
        ax.plot(offset[counter], np.mean(progressive_saccades.saccade_amplitude), marker=1, markersize=10,
                color=colors_p[counter], alpha=1.0)

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100),
                              label=[input_noise_magnitude[0], n_saccades[0], n_progressive_saccades[0]]),
               mpatches.Patch(facecolor=plt.cm.Greens(100),
                              label=[input_noise_magnitude[1], n_saccades[1], n_progressive_saccades[1]]),
               mpatches.Patch(facecolor=plt.cm.Blues(100),
                              label=[input_noise_magnitude[2], n_saccades[2], n_progressive_saccades[2]])]
    ax.legend(handles=handles)

    if safe_plot:
        plt.savefig(f"{path_to_save_folder}saccade_amplitude_level_{level}_drift_enabled_{drift_enabled}", dpi=300)
        plt.close()


def plot_saccade_vectors(eye_data_none, eye_data_weak, eye_data_strong, level, drift_enabled, safe_plot=False,
                         path_to_save_folder=f"{os.getcwd()}/plots/plots_saccade_vectors/"):
    """
    :param eye_data_none: data of onput noise = nan
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 3 subplots
    fig.suptitle(f"saccade vectors; level {level}, drift_enabled = {drift_enabled}")

    fig.supxlabel("observation space x")
    fig.supylabel("observation space y")

    # Plotting
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["NaN", "weak", "strong"]
    n_saccades = [np.nan, np.nan, np.nan]
    n_progressive_saccades = [np.nan, np.nan, np.nan]

    counter = 0

    for eye_data, ax in zip([eye_data_none, eye_data_weak, eye_data_strong], axs.ravel()):

        # axis labels
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")

        ax.set_xlim([-1000, 1000])
        ax.set_ylim([-600, 600])

        # all kinds of saccades
        saccades = eye_data[eye_data["saccadeOnset"] >= 1]
        n_saccades[counter] = len(saccades)
        # progressive saccades
        progressive_saccades = saccades.loc[saccades["saccade_direction_y"] <= 0]
        n_progressive_saccades[counter] = len(progressive_saccades)

        # loop through saccades
        for saccade in np.arange(len(progressive_saccades)):
            saccade_launch_site = [progressive_saccades.iloc[saccade].converging_eye_x,
                                   progressive_saccades.iloc[saccade].converging_eye_y]
            saccade_landing_site = [progressive_saccades.iloc[saccade].converging_eye_x + progressive_saccades.iloc[saccade].saccade_direction_x,
                                    progressive_saccades.iloc[saccade].converging_eye_y + progressive_saccades.iloc[saccade].saccade_direction_y]

            x_pos = [saccade_launch_site[0], saccade_landing_site[0]]
            y_pos = [saccade_launch_site[1], saccade_landing_site[1]]

            # draw on canvas
            ax.quiver(x_pos, y_pos, progressive_saccades.iloc[saccade].saccade_direction_x,
                      progressive_saccades.iloc[saccade].saccade_direction_y, scale_units='xy', angles='xy', scale=1,
                      color=colors[counter], alpha=1.0)

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100),
                              label=[input_noise_magnitude[0], n_progressive_saccades[0], n_saccades[0]]),
               mpatches.Patch(facecolor=plt.cm.Greens(100),
                              label=[input_noise_magnitude[1], n_progressive_saccades[1], n_saccades[1]]),
               mpatches.Patch(facecolor=plt.cm.Blues(100),
                              label=[input_noise_magnitude[2], n_progressive_saccades[2], n_saccades[2]])]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.9), framealpha=1.0)

    if safe_plot:
        plt.savefig(f"{path_to_save_folder}saccade_vectors_level_{level}_drift_enabled_{drift_enabled}", dpi=300)
        plt.close()


def plot_saccade_landing_sites(eye_data_none, eye_data_weak, eye_data_strong, level, drift_enabled, safe_plot=False,
                               path_to_save_folder=f"{os.getcwd()}/plots/plots_saccade_landing_site/"):
    """
    :param eye_data_none: data of onput noise = nan
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 3 subplots
    fig.suptitle(f"saccade landing sites; level {level}, drift_enabled = {drift_enabled}")

    fig.supxlabel("observation space x")
    fig.supylabel("observation space y")

    # Plotting
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["NaN", "weak", "strong"]
    n_saccades = [np.nan, np.nan, np.nan]
    n_progressive_saccades = [np.nan, np.nan, np.nan]

    counter = 0

    for eye_data, ax in zip([eye_data_none, eye_data_weak, eye_data_strong], axs.ravel()):
        # axis labels
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")

        ax.set_xlim([-1000, 1000])
        ax.set_ylim([-600, 400])

        # all kinds of saccades
        saccades = eye_data[eye_data["saccadeOnset"] >= 1]
        n_saccades[counter] = len(saccades)
        # progressive saccades
        progressive_saccades = saccades.loc[saccades["saccade_direction_y"] <= 0]
        n_progressive_saccades[counter] = len(progressive_saccades)
        ## insert saccade landing sites
        progressive_saccades["saccade_landing_site_x"] = progressive_saccades.converging_eye_x + progressive_saccades.saccade_direction_x
        progressive_saccades["saccade_landing_site_y"] = progressive_saccades.converging_eye_y + progressive_saccades.saccade_direction_y

        # draw on canvas
        heatmap = sns.kdeplot(x=progressive_saccades.saccade_landing_site_x,
                              y=progressive_saccades.saccade_landing_site_y,
                              cmap=color_maps[counter],
                              shade=True,
                              alpha=0.9,
                              bw_adjust=0.4,
                              ax=ax)

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100),
                              label=[input_noise_magnitude[0], n_progressive_saccades[0], n_saccades[0]]),
               mpatches.Patch(facecolor=plt.cm.Greens(100),
                              label=[input_noise_magnitude[1], n_progressive_saccades[1], n_saccades[1]]),
               mpatches.Patch(facecolor=plt.cm.Blues(100),
                              label=[input_noise_magnitude[2], n_progressive_saccades[2], n_saccades[2]])]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.9), framealpha=1.0)

    if safe_plot:
        plt.savefig(f"{path_to_save_folder}saccade_landing_site_level_{level}_drift_enabled_{drift_enabled}", dpi=300)
        plt.close()