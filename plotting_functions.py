import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import scipy.stats as st
import os
import matplotlib.patches as mpatches
import seaborn as sns
from helper_functions import point_estimate


def plot_kde_combined(input_data, eye_data, safe_plot=True, path_to_save_folder=f"{os.getcwd()}/plots/kde_plots_event_distribution/"):

    # reducing data to only respective events of interest
    inputs = input_data[input_data["start_input"] == 1.0]
    saccades = eye_data[eye_data["saccadeOnset"] == 1.0]

    # extract level features
    level = inputs.iloc[0].trial
    drift_enabled = inputs.iloc[0].drift_enabled
    input_noise = inputs.iloc[0].input_noise_magnitude

    # define arrays of time tags for respective data
    input_data_array = np.asarray(inputs.time_played)

    # define point dataframes
    input_data_points = {'x': input_data_array, 'y': [0]*len(input_data_array)}
    input_data_points = pd.DataFrame(data=input_data_points)
    # dataframe for eye_data points (pro & regressive) is generated down below...

    # compute hpdi (I went for the smallest interval which contains 25% of the mass)
    input_data_hpdi_bounds = az.hdi(input_data_array, 0.25)


    # plot boundaries:
    lbound = 0
    ubound = input_data.iloc[-2].time_played  # second last row because last row is written to df AFTER SoC response given which may take time


    # instatiate KDEs
    kde_init = np.linspace(lbound, ubound, 100)

    input_data_kde = st.gaussian_kde(input_data_array)


    # Grid
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Densities of events for level {level} (drift enabled = {drift_enabled}, input noise = {input_noise}, crashed = {input_data.iloc[-1].collision})", fontdict={"fontweight": "bold"})


    # axis labels
    ax.set_xlabel("Time played")
    ax.set_ylabel("Density")

    ax.set_xlim([lbound, ubound])

    xaxis = np.linspace(lbound, ubound, 10)
    ax.set_xticks(xaxis)


    # Plotting
    colors = ["crimson", "limegreen", "royalblue"]

    ax.plot(kde_init, input_data_kde(kde_init), color=colors[0], label='keyboard inputs')
    #ax.fill_between(kde_init, input_data_kde(kde_init), step="mid", alpha=0.3, color=colors[0])
    ax.scatter(input_data_points.x, input_data_points.y, marker=".", color=colors[0])

    y_max = ax.get_ylim()[1]  # 0: bottom; 1: top


    # point estimates and HPDIs:
    point_estimate_input_data = point_estimate(inputs.time_played)
    #ax.axvspan(point_estimate_input_data[2], point_estimate_input_data[3], alpha=0.3, color=colors[0])
    plt.vlines(point_estimate_input_data[0], ymin=0, ymax=point_estimate_input_data[1], color=colors[0])

    # eye-movement behavior
    # progressive saccades
    progressive_saccades = saccades.loc[saccades["saccade_direction_y"] < 0]
    progressive_eye_data_array = np.asarray(progressive_saccades.time_tag)

    progressive_eye_data_hpdi_bounds = az.hdi(progressive_eye_data_array, 0.25)
    progressive_eye_data_kde = st.gaussian_kde(progressive_eye_data_array)
    ax.plot(kde_init, progressive_eye_data_kde(kde_init), color=colors[1], label='progressive eye movements')

    progressive_eye_data_points = {'x': progressive_eye_data_array, 'y': [y_max/95]*len(progressive_eye_data_array)}
    progressive_eye_data_points = pd.DataFrame(data=progressive_eye_data_points)
    ax.scatter(progressive_eye_data_points.x, progressive_eye_data_points.y, marker=".", color=colors[1])

    point_estimate_progressive_eye_data = point_estimate(progressive_saccades.time_tag)
    plt.vlines(point_estimate_progressive_eye_data[0], ymin=0, ymax=point_estimate_progressive_eye_data[1], color=colors[1])

    # regressive saccades
    regressive_saccades = saccades.loc[saccades["saccade_direction_y"] >= 0]
    regressive_eye_data_array = np.asarray(regressive_saccades.time_tag)

    regressive_eye_data_hpdi_bounds = az.hdi(regressive_eye_data_array, 0.25)
    regressive_eye_data_kde = st.gaussian_kde(regressive_eye_data_array)
    ax.plot(kde_init, regressive_eye_data_kde(kde_init), color=colors[2], label='regressive eye movements')

    regressive_eye_data_points = {'x': regressive_eye_data_array, 'y': [y_max/95]*len(regressive_eye_data_array)}
    regressive_eye_data_points = pd.DataFrame(data=regressive_eye_data_points)
    ax.scatter(regressive_eye_data_points.x, regressive_eye_data_points.y, marker=".", color=colors[2])

    point_estimate_regressive_eye_data = point_estimate(regressive_saccades.time_tag)
    plt.vlines(point_estimate_regressive_eye_data[0], ymin=0, ymax=point_estimate_regressive_eye_data[1], color=colors[2])

    ax.legend()

    if safe_plot:
        plt.savefig(f"{path_to_save_folder}Event_densities_level_{level}_drift_enabled_{drift_enabled}_input_noise_{input_noise}", dpi=300)
        plt.close()


def plot_fixation_location_kde(eye_data_none, eye_data_weak, eye_data_strong, level=1, drift_enabled=False,
                               safe_plot=False,
                               path_to_save_folder=f"{os.getcwd()}/plots/kde_plots_fixation_locations/"):
    """
    :param eye_data_none: data of onput noise = none
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
    input_noise_magnitude = ["none", "weak", "strong"]
    counter = 0

    for eye_data, ax in zip([eye_data_none, eye_data_weak, eye_data_strong], axs.ravel()):
        # axis labels
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")

        ax.set_xlim([-300, 300])
        ax.set_ylim([-600, 600])

        fixations = eye_data[eye_data["fixationOnset"] == 1.0]
        # check for fixation within game boarders
        fixations = fixations[fixations["converging_eye_x_adjusted"].between((34 * 18), ((34 + 40) * 18))]
        # edge=34, scaling=18, observation_space_x=40

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


def plot_fixation_duration(eye_data_none, eye_data_weak, eye_data_strong, level, drift_enabled,
                           exploring_fixations=True, safe_plot=False,
                           path_to_save_folder=f"{os.getcwd()}/plots/plota_fixation_duration/"):
    """
    :param eye_data_none: data of onput noise = none
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param exploring_fixations: True vs. False; if True progressive fixations are considered, else resting fixations
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """

    if exploring_fixations:
        fixation_type = "exploring_fixations"
    else:
        fixation_type = "resting_fixations"

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title(f"fixation durations - {fixation_type}; level {level}, drift_enabled = {drift_enabled}",
                 fontdict={"fontweight": "bold"})

    # axis labels
    ax.set_xlabel("input noise magnitude")
    ax.set_ylabel("fixation duration (in ms)")

    ax.set_xlim([-70, 70])
    ax.set_xticks([-50, 0, 50])
    ax.set_xticklabels(["none", "weak", "strong"])

    # Plotting
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["none", "weak", "strong"]
    offset = [-50, 0, 50]
    n_fixations = [np.nan, np.nan, np.nan]
    n_target_fixations = [np.nan, np.nan, np.nan]

    counter = 0

    for eye_data in [eye_data_none, eye_data_weak, eye_data_strong]:

        # all kinds of saccades
        # only consider the ones whichs amplitude is below 800 (arbitrarily chosen - saccades out of screen)
        fixations = eye_data[eye_data["fixationOnset"] >= 1]
        n_fixations[counter] = len(fixations)
        plot_labels = [offset[counter]] * len(fixations)
        fixations["plot_label"] = [offset[counter]] * len(fixations)
        # subset target fixations
        if exploring_fixations:
            target_fixations = fixations.loc[fixations["exploring_fixation"] == 1]
        else:
            target_fixations = fixations.loc[fixations["exploring_fixation"] == 0]
        n_target_fixations[counter] = len(target_fixations)

        # draw on canvas
        ax.scatter(target_fixations.plot_label, target_fixations.fixation_duration, marker=".", color=colors_p[counter],
                   alpha=0.3)
        ax.plot(offset[counter], np.mean(target_fixations.fixation_duration), marker="_", markersize=15,
                color=colors_p[counter], alpha=1.0)

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100),
                              label=[input_noise_magnitude[0], n_target_fixations[0], n_fixations[0]]),
               mpatches.Patch(facecolor=plt.cm.Greens(100),
                              label=[input_noise_magnitude[1], n_target_fixations[1], n_fixations[1]]),
               mpatches.Patch(facecolor=plt.cm.Blues(100),
                              label=[input_noise_magnitude[2], n_target_fixations[2], n_fixations[2]])]
    ax.legend(handles=handles)

    if safe_plot:
        plt.savefig(
            f"{path_to_save_folder}fixation_duration_{fixation_type}_level_{level}_drift_enabled_{drift_enabled}",
            dpi=300)
        plt.close()


def plot_eye_rest_y_over_time(eye_data_none, eye_data_weak, eye_data_strong, input_data, safe_plot=False,
                              path_to_save_folder=f"{os.getcwd()}/plots/eye_resting_position_y/"):
    """
    :param eye_data_none: data of onput noise = none
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
    ax.set_title(f"eye resting y positions; level = {level}, drift_enabled = {drift_enabled}",
                 fontdict={"fontweight": "bold"})

    # axis labels
    ax.set_xlabel("time_played")
    ax.set_ylabel("observation space y")

    ax.set_ylim([-200, 500])

    # plt.gca().invert_yaxis()

    # Plotting
    ## initiate colors and labels
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["none", "weak", "strong"]

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


def plot_saccade_amplitudes(eye_data_none, eye_data_weak, eye_data_strong, level, drift_enabled,
                            regressive_saccades=False, safe_plot=False,
                            path_to_save_folder=f"{os.getcwd()}/plots/plots_saccade_amplitude/"):
    """
    :param eye_data_none: data of onput noise = none
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param regressive_saccades: True vs. False; if True regressive saccades are targeted else progressive saccades
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """

    if regressive_saccades:
        saccade_type = "regressive_saccades"
    else:
        saccade_type = "progressive_saccades"

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title(f"saccade amplitudes - {saccade_type}; level {level}, drift_enabled = {drift_enabled}",
                 fontdict={"fontweight": "bold"})

    # axis labels
    ax.set_xlabel("input noise magnitude")
    ax.set_ylabel("saccade amplitude (in °)")

    ax.set_xlim([-70, 70])
    ax.set_xticks([-50, 0, 50])
    ax.set_xticklabels(["none", "weak", "strong"])

    # Plotting
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["none", "weak", "strong"]
    offset = [-50, 0, 50]
    n_saccades = [np.nan, np.nan, np.nan]
    n_target_saccades = [np.nan, np.nan, np.nan]

    counter = 0

    for eye_data in [eye_data_none, eye_data_weak, eye_data_strong]:

        # all kinds of saccades
        # only consider the ones whichs amplitude is below 800 (arbitrarily chosen - saccades out of screen)
        saccades = eye_data[(eye_data["saccadeOnset"] >= 1) & (eye_data.saccade_amplitude < 800)]
        n_saccades[counter] = len(saccades)
        plot_labels = [offset[counter]] * len(saccades)
        saccades["plot_label"] = [offset[counter]] * len(saccades)
        # subset target saccades
        if regressive_saccades:
            target_saccades = saccades.loc[saccades["saccade_direction_y"] >= 0]
        else:
            target_saccades = saccades.loc[saccades["saccade_direction_y"] < 0]
        n_target_saccades[counter] = len(target_saccades)

        # draw on canvas
        ax.scatter(target_saccades.plot_label, target_saccades.saccade_amplitude, marker=".", color=colors_p[counter],
                   alpha=0.3)
        ax.plot(offset[counter], np.mean(target_saccades.saccade_amplitude), marker="_", markersize=15,
                color=colors_p[counter], alpha=1.0)

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100),
                              label=[input_noise_magnitude[0], n_target_saccades[0], n_saccades[0]]),
               mpatches.Patch(facecolor=plt.cm.Greens(100),
                              label=[input_noise_magnitude[1], n_target_saccades[1], n_saccades[1]]),
               mpatches.Patch(facecolor=plt.cm.Blues(100),
                              label=[input_noise_magnitude[2], n_target_saccades[2], n_saccades[2]])]
    ax.legend(handles=handles)

    if safe_plot:
        plt.savefig(
            f"{path_to_save_folder}saccade_amplitude_{saccade_type}_level_{level}_drift_enabled_{drift_enabled}",
            dpi=300)
        plt.close()


def plot_saccade_vectors(eye_data_none, eye_data_weak, eye_data_strong, level, drift_enabled, regressive_saccades=False,
                         safe_plot=False, path_to_save_folder=f"{os.getcwd()}/plots/plots_saccade_vectors/"):
    """
    :param eye_data_none: data of onput noise = none
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param regressive_saccades: True vs. False; if True regressive saccades are targeted else progressive saccades
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """

    if regressive_saccades:
        saccade_type = "regressive_saccades"
    else:
        saccade_type = "progressive_saccades"

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 3 subplots
    fig.suptitle(f"saccade vectors - {saccade_type}; level {level}, drift_enabled = {drift_enabled}")

    fig.supxlabel("observation space x")
    fig.supylabel("observation space y")

    # Plotting
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["none", "weak", "strong"]
    n_saccades = [np.nan, np.nan, np.nan]
    n_target_saccades = [np.nan, np.nan, np.nan]

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

        # subset target saccades
        if regressive_saccades:
            target_saccades = saccades.loc[saccades["saccade_direction_y"] >= 0]
        else:
            target_saccades = saccades.loc[saccades["saccade_direction_y"] < 0]

        n_target_saccades[counter] = len(target_saccades)

        # loop through saccades
        for saccade in np.arange(len(target_saccades)):
            saccade_launch_site = [target_saccades.iloc[saccade].converging_eye_x,
                                   target_saccades.iloc[saccade].converging_eye_y]
            saccade_landing_site = [
                target_saccades.iloc[saccade].converging_eye_x + target_saccades.iloc[saccade].saccade_direction_x,
                target_saccades.iloc[saccade].converging_eye_y + target_saccades.iloc[saccade].saccade_direction_y]

            x_pos = [saccade_launch_site[0], saccade_landing_site[0]]
            y_pos = [saccade_launch_site[1], saccade_landing_site[1]]

            # draw on canvas
            ax.quiver(x_pos, y_pos, target_saccades.iloc[saccade].saccade_direction_x,
                      target_saccades.iloc[saccade].saccade_direction_y, scale_units='xy', angles='xy', scale=1,
                      color=colors[counter], alpha=1.0)

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100),
                              label=[input_noise_magnitude[0], n_target_saccades[0], n_saccades[0]]),
               mpatches.Patch(facecolor=plt.cm.Greens(100),
                              label=[input_noise_magnitude[1], n_target_saccades[1], n_saccades[1]]),
               mpatches.Patch(facecolor=plt.cm.Blues(100),
                              label=[input_noise_magnitude[2], n_target_saccades[2], n_saccades[2]])]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.9), framealpha=1.0)

    if safe_plot:
        plt.savefig(f"{path_to_save_folder}saccade_vectors_{saccade_type}_level_{level}_drift_enabled_{drift_enabled}",
                    dpi=300)
        plt.close()


def plot_saccade_landing_sites(eye_data_none, eye_data_weak, eye_data_strong, level, drift_enabled,
                               regressive_saccades=False, safe_plot=False,
                               path_to_save_folder=f"{os.getcwd()}/plots/plots_saccade_landing_site/"):
    """
    :param eye_data_none: data of onput noise = none
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param regressive_saccades: True vs. False; if True regressive saccades are targeted else progressive saccades
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """
    if regressive_saccades:
        saccade_type = "regressive_saccades"
    else:
        saccade_type = "progressive_saccades"

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 3 subplots
    fig.suptitle(f"saccade landing sites - {saccade_type}; level {level}, drift_enabled = {drift_enabled}")

    fig.supxlabel("observation space x")
    fig.supylabel("observation space y")

    # Plotting
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["none", "weak", "strong"]
    n_saccades = [np.nan, np.nan, np.nan]
    n_target_saccades = [np.nan, np.nan, np.nan]

    counter = 0

    for eye_data, ax in zip([eye_data_none, eye_data_weak, eye_data_strong], axs.ravel()):

        # axis labels
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")

        ax.set_xlim([-1500, 1500])
        ax.set_ylim([-800, 400])

        # all kinds of saccades
        saccades = eye_data[eye_data["saccadeOnset"] >= 1]
        n_saccades[counter] = len(saccades)
        # subset target saccades
        if regressive_saccades:
            target_saccades = saccades.loc[saccades["saccade_direction_y"] >= 0]
        else:
            target_saccades = saccades.loc[saccades["saccade_direction_y"] < 0]
        n_target_saccades[counter] = len(target_saccades)
        # insert saccade landing sites
        target_saccades[
            "saccade_landing_site_x"] = target_saccades.converging_eye_x + target_saccades.saccade_direction_x
        target_saccades[
            "saccade_landing_site_y"] = target_saccades.converging_eye_y + target_saccades.saccade_direction_y

        # filter for landing sites within level
        target_saccades = target_saccades[
            target_saccades["saccade_landing_site_x"].between(((34 - 20) * 18), ((34 + 40 + 20) * 18))]
        # edge=34, scaling=18, observation_space_x=40

        # draw on canvas
        heatmap = sns.kdeplot(x=target_saccades.saccade_landing_site_x,
                              y=target_saccades.saccade_landing_site_y,
                              cmap=color_maps[counter],
                              shade=True,
                              alpha=0.9,
                              bw_adjust=0.4,
                              ax=ax)

        # y_coord = point_estimate(progressive_saccades.saccade_landing_site_y)[0]
        # ax.axhline(y_coord, color=colors[counter])
        # x_coord = point_estimate(progressive_saccades.saccade_landing_site_x)[0]
        # ax.axvline(x_coord, color=colors[counter])

        counter += 1

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100),
                              label=[input_noise_magnitude[0], n_target_saccades[0], n_saccades[0]]),
               mpatches.Patch(facecolor=plt.cm.Greens(100),
                              label=[input_noise_magnitude[1], n_target_saccades[1], n_saccades[1]]),
               mpatches.Patch(facecolor=plt.cm.Blues(100),
                              label=[input_noise_magnitude[2], n_target_saccades[2], n_saccades[2]])]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.9), framealpha=1.0)

    if safe_plot:
        plt.savefig(
            f"{path_to_save_folder}saccade_landing_site_{saccade_type}_level_{level}_drift_enabled_{drift_enabled}",
            dpi=300)
        plt.close()


def plot_saccade_amplitude_kde(eye_data_none, eye_data_weak, eye_data_strong, level, drift_enabled,
                               regressive_saccades=False, safe_plot=False,
                               path_to_save_folder=f"{os.getcwd()}/plots/plots_saccade_amplitude_kde/"):
    """
    :param eye_data_none: data of onput noise = none
    :param eye_data_weak: data of onput noise = weak
    :param eye_data_strong: data of onput noise = strong
    :param level: level played
    :param drift_enabled: True vs. False
    :param regressive_saccades: True vs. False; if True regressive saccades are targeted else progressive saccades
    :param safe_plot: True vs. False - determines whether plots are saved or not
    :param path_to_save_folder: path to the folder in which plots are saved in case of safe_plot=True
    IMPORTANT: The function will not raise an error if the runs aren't of the same level and drift condition. It will only consider level and drift passed for labeling
    """
    if regressive_saccades:
        saccade_type = "regressive_saccades"
    else:
        saccade_type = "progressive_saccades"
    # Grid
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Densities of saccade amplitudes - {saccade_type}; level {level} (drift enabled = {drift_enabled})",
                 fontdict={"fontweight": "bold"})

    # axis labels
    ax.set_xlabel("saccade amplitude")
    ax.set_ylabel("Density")

    # plot boundaries:
    lbound = 0
    ubound = 1000

    ax.set_xlim([lbound, ubound])

    xaxis = np.linspace(lbound, ubound, 11)
    ax.set_xticks(xaxis)

    # Plotting parameters
    colors = ["coral", "lightgreen", "royalblue"]
    colors_p = ["crimson", "green", "blue"]
    color_maps = ["Reds", "Greens", "Blues"]
    input_noise_magnitude = ["none", "weak", "strong"]
    n_saccades = [np.nan, np.nan, np.nan]
    n_target_saccades = [np.nan, np.nan, np.nan]

    counter = 0

    for temp_eye_data in [eye_data_none, eye_data_weak, eye_data_strong]:
        # reducing data to only respective events of interest (saccade onsets) & filter enormous saccade amplitudes
        saccades = temp_eye_data[(temp_eye_data["saccadeOnset"] == 1.0) & (temp_eye_data["saccade_amplitude"] < 1000)]

        # subset target saccades
        if regressive_saccades:
            saccades = saccades.loc[saccades["saccade_direction_y"] >= 0]
        else:
            saccades = saccades.loc[saccades["saccade_direction_y"] < 0]

        n_saccades[counter] = len(saccades)
        ## insert saccade landing sites
        saccades["saccade_landing_site_x"] = saccades.converging_eye_x + saccades.saccade_direction_x
        saccades["saccade_landing_site_y"] = saccades.converging_eye_y + saccades.saccade_direction_y

        # filter for landing sites within level
        saccades = saccades[saccades["saccade_landing_site_x"].between(((34 - 20) * 18), ((34 + 40 + 20) * 18))]
        # edge=34, scaling=18, observation_space_x=40

        # define arrays of saccade amplitudes for respective data
        eye_data_array = np.asarray(saccades.saccade_amplitude)

        # compute hpdi (I went for the smallest interval which contains 25% of the mass)
        eye_data_hpdi_bounds = az.hdi(eye_data_array, 0.25)

        # instatiate KDEs
        kde_init = np.linspace(lbound, ubound, 100)

        eye_data_kde = st.gaussian_kde(eye_data_array)

        # draw KDE

        ax.plot(kde_init, eye_data_kde(kde_init), color=colors[counter], label=input_noise_magnitude[counter])

        # define dataframe for points of eye_data ( now because we can retrieve y-axis limits at this point)
        # y_max = ax.get_ylim()[1]  # 0: bottom; 1: top
        # eye_data_points = {'x': eye_data_array, 'y': [y_max/99]*len(eye_data_array)}  # y_max/99 to plot these a bit above input data points
        # eye_data_points = pd.DataFrame(data=eye_data_points)
        # ax.scatter(eye_data_points.x, eye_data_points.y, marker=".", color=colors[counter])

        # HPDIs:
        point_estimate_eye_data = point_estimate(saccades.time_tag)
        # plt.vlines(point_estimate_eye_data[0], ymin=0, ymax=point_estimate_eye_data[1], color=colors[counter])

        counter += 1

    ax.legend()

    if safe_plot:
        plt.savefig(
            f"{path_to_save_folder}saccade_amplitude_kde_{saccade_type}_level_{level}_drift_enabled_{drift_enabled}",
            dpi=300)
        plt.close()
