import numpy as np
import pandas as pd
# import seaborn as sns
# import pymc3
import arviz as az
import scipy.stats as st
import matplotlib.pyplot as plt
from ast import literal_eval


def pre_process_input_data(dataframe):
    """
    dataframe must be pandas dataFrame with appropiate columns...
    """

    dataframe.rename(columns={'Unnamed: 0': 'frame'}, inplace=True)

    # convert columns with literal_eval to not have strings anymore
    dataframe.player_pos = dataframe.player_pos.apply(lambda row: literal_eval(row))
    dataframe.visible_obstacles = dataframe.visible_obstacles.apply(lambda row: literal_eval(row))
    dataframe.visible_drift_tiles = dataframe.visible_drift_tiles.apply(lambda row: literal_eval(row))

    # adjust time tag
    dataframe['adjusted_time_tag'] = dataframe.time_played + dataframe.time_tag

    # annotate input data #

    # input can be either None, "Right", "Left"
    rows_with_input = dataframe[~dataframe["current_input"].isnull()][["frame"]]
    # subsetting columns

    # condition for start input
    cond = rows_with_input.frame - rows_with_input.frame.shift(1) >= 2

    # have =1 everywhere condition applies and =0 where not
    rows_with_input["start_input"] = np.where(cond, 1, 0)

    # flag first row also as start input (because we missed it by not taking any rows before that row due to subsetting)
    index = rows_with_input.iloc[0].frame
    rows_with_input.loc[index, "start_input"] = 1.0

    # label all frames of each individual input with number of input
    rows_with_input["N_input"] = (rows_with_input["start_input"] == 1).cumsum()

    # joining dataframes
    dataframe = dataframe.merge(rows_with_input, on="frame", how='left')

    # annotate input switch (strategy change?) #
    rows_with_input_direction = dataframe[dataframe["start_input"] == 1][["frame", "current_input"]]

    # condition for input switch
    cond = rows_with_input_direction.current_input != rows_with_input_direction.current_input.shift(1)

    # have =1 everywhere condition applies and =0 where not
    rows_with_input_direction["input_change"] = np.where(cond, 1, 0)
    # drop current_input column for better merge in next step
    rows_with_input_direction.drop(columns="current_input", axis=1, inplace=True)

    # joining dataframes
    dataframe = dataframe.merge(rows_with_input_direction, on="frame", how='left')

    return dataframe


def calc_saccade_direction(saccade_rows):
    """
    Need to be given pandas dataframe grouped by N_saccade.
    Dataframe must have columns of eye positions (e.g. LeftEyeX) and saccade_direction.
    Returned dataframe now holds direction vector of saccade in every row in column saccade direction.
    """

    # x-direction
    left_eye_x_direction = saccade_rows.iloc[-1].LeftEyeX - saccade_rows.iloc[0].LeftEyeX
    right_eye_x_direction = saccade_rows.iloc[-1].RightEyeX - saccade_rows.iloc[0].RightEyeX
    x_direction = (left_eye_x_direction + right_eye_x_direction) / 2

    # y-direction
    left_eye_y_direction = saccade_rows.iloc[-1].LeftEyeY - saccade_rows.iloc[0].LeftEyeY
    right_eye_y_direction = saccade_rows.iloc[-1].RightEyeY - saccade_rows.iloc[0].RightEyeY
    y_direction = (left_eye_y_direction + right_eye_y_direction) / 2

    saccade_rows.saccade_direction = saccade_rows.apply(lambda x: [x_direction, y_direction], axis=1)

    return saccade_rows


def pre_process_eye_data(eye_data):
    """
    annotate eye_tracking data
    """

    # adjust time tag to start at 0
    eye_data["time_tag"] = eye_data.TimeTag - eye_data.TimeTag[0]

    # annotate binocular saccades
    eye_data["Saccade"] = eye_data.LeftEyeSaccadeFlag + eye_data.RightEyeSaccadeFlag
    # eliminate simultaneous blink and saccades (setting saccade to 0) #
    eye_data.Saccade.loc[eye_data.LeftBlink > 0.0] = 0.0
    eye_data.Saccade.loc[eye_data.RightBlink > 0.0] = 0.0

    eye_data.Saccade[eye_data.Saccade > 1] = 1.0

    # condition for initiating saccade
    cond = (eye_data.Saccade >= 1.0) & (eye_data.Saccade.shift(1) == 0.0)

    # have =1 everywhere condition applies and =0 where not
    eye_data["saccadeOnset"] = np.where(cond, 1, 0)

    # insert N_saccade - counting up saccades
    eye_data["N_saccade"] = (eye_data["saccadeOnset"] == 1).cumsum()
    eye_data.loc[eye_data.Saccade < 1.0, "N_saccade"] = np.nan  # have NaN everywhere where there is no saccade

    # insert saccade direction column
    eye_data["saccade_direction"] = np.nan
    eye_data = eye_data.groupby("N_saccade", group_keys=True).apply(calc_saccade_direction)

    return eye_data


def point_estimate(data):
    """
    function for estimating point of maximum for kde
    """

    kde = st.gaussian_kde(data)  # gaussian kernel
    n_samples = 1000  # arbitrarily high number of samples
    samples = np.linspace(min(data), max(data), n_samples)  # sampling
    probs = kde.evaluate(samples)
    point_estimate_y = max(probs)
    point_estimate_index = probs.argmax()
    point_estimate_x = samples[point_estimate_index]
    hdi = az.hdi(samples, hdi_prob=0.25)  # compute hpdi (I went for the smallest interval which contains 25% of the mass)

    return point_estimate_x, point_estimate_y, hdi[0], hdi[1]


def plot_kde_combined(code="pilot4", n_run=0, safe_plot=True):
    # load data (with input data being button presses and eye data being eye tracking output)
    input_data = pd.read_csv(f'input_data/{code}_output_{n_run:0>2}.csv', index_col=False)
    input_data = pre_process_input_data(input_data)

    eye_data = pd.read_csv(f'eye_data/{code}_eye_tracking_output_{n_run:0>2}.csv', index_col=False)
    eye_data = pre_process_eye_data(eye_data)

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
    colors = ["crimson", "limegreen"]

    ax.plot(kde_init, input_data_kde(kde_init), color=colors[0], label='input data')
    # ax.fill_between(kde_init, input_data_kde(kde_init), step="mid", alpha=0.3, color=colors[0])
    ax.scatter(input_data_points.x, input_data_points.y, marker=".", color=colors[0])

    ax.plot(kde_init, eye_data_kde(kde_init), color=colors[1], label='eye movement data')
    # ax.fill_between(kde_init, eye_data_kde(kde_init), step="mid", alpha=0.3, color=colors[1])

    # define dataframe for points of eye_data ( now because we can retrieve y-axis limits at this point)
    y_max = ax.get_ylim()[1]  # 0: bottom; 1: top
    eye_data_points = {'x': eye_data_array, 'y': [y_max / 99] * len(eye_data_array)}  # y_max/99 to plot these a bit above input data points
    eye_data_points = pd.DataFrame(data=eye_data_points)
    ax.scatter(eye_data_points.x, eye_data_points.y, marker=".", color=colors[1])

    # HPDIs:
    point_estimate_input_data = point_estimate(inputs.time_played)
    # ax.axvspan(point_estimate_input_data[2], point_estimate_input_data[3], alpha=0.3, color=colors[0])
    plt.vlines(point_estimate_input_data[0], ymin=0, ymax=point_estimate_input_data[1], color=colors[0])

    point_estimate_eye_data = point_estimate(saccades.time_tag)
    # ax.axvspan(point_estimate_eye_data[2], point_estimate_eye_data[3], alpha=0.3, color=colors[1])
    plt.vlines(point_estimate_eye_data[0], ymin=0, ymax=point_estimate_eye_data[1], color=colors[1])

    ax.legend()

    if safe_plot:
        plt.savefig(f"kde_plots/Event densities trial {inputs.iloc[0].trial} run {n_run}", dpi=300)
    plt.close()
