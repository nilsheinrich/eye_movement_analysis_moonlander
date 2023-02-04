import numpy as np
import pandas as pd
from ast import literal_eval
import arviz as az
import scipy.stats as st
pd.options.mode.chained_assignment = None  # default='warn'


def pre_process_input_data(dataframe):
    """
    dataframe must be pandas dataFrame with appropriate columns...
    """

    dataframe.rename(columns={'Unnamed: 0': 'frame'}, inplace=True)

    # convert columns with literal_eval to not have strings anymore
    dataframe.player_pos = dataframe.player_pos.apply(lambda row: literal_eval(row))
    dataframe.visible_obstacles = dataframe.visible_obstacles.apply(lambda row: literal_eval(row))
    dataframe.visible_drift_tiles = dataframe.visible_drift_tiles.apply(lambda row: literal_eval(row))

    # adjust time tag
    dataframe['adjusted_time_tag'] = dataframe.time_played + dataframe.time_tag

    ## annotate input data

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

    ## annotate input switch (strategy change?)
    rows_with_input_direction = dataframe[dataframe["start_input"] == 1][["frame", "current_input"]]

    # condition for input switch
    cond = rows_with_input_direction.current_input != rows_with_input_direction.current_input.shift(1)

    # have =1 everywhere condition applies and =0 where not
    rows_with_input_direction["input_change"] = np.where(cond, 1, 0)
    # drop current_input column for better merge in next step
    rows_with_input_direction.drop(columns="current_input", axis=1, inplace=True)

    # joining dataframes
    dataframe = dataframe.merge(rows_with_input_direction, on="frame", how='left')

    # flagging drift onset (and second drift onset)
    # condition for drift onset
    cond = (dataframe["visible_drift_tiles"].str.len() != 0) & (
                dataframe["visible_drift_tiles"].shift(1).str.len() == 0)

    # have =1 everywhere condition applies and =0 where not
    dataframe["drift_tile_onset"] = np.where(cond, 1, 0)

    # condition for multiple drift tiles on screen
    cond = (dataframe["visible_drift_tiles"].shift(1).str.len() != 0) & (
                dataframe["visible_drift_tiles"].str.len() > dataframe["visible_drift_tiles"].shift(1).str.len())

    # have =1 everywhere condition applies and =0 where not
    dataframe["second_drift_tile_onset"] = np.where(cond, 1, 0)

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

    saccade_rows.saccade_direction_x = saccade_rows.apply(lambda x: x_direction, axis=1)
    saccade_rows.saccade_direction_y = saccade_rows.apply(lambda x: y_direction, axis=1)
    saccade_rows.saccade_amplitude = saccade_rows.apply(
        lambda x: np.sqrt(np.power(x_direction, 2) + np.power(y_direction, 2)), axis=1)

    return saccade_rows


# annotate eye_tracking data

def pre_process_eye_data(eye_data, screen_width_in_mm=595, screen_height_in_mm=335, pixels_width=1920,
                         pixels_height=1080, distance_to_screen_in_mm=770):
    """
    dataframe must be pandas dataFrame with appropriate columns...
    """

    # calc how many pixels are within 1 mm on screen
    pixels_in_mm = ((pixels_width / screen_width_in_mm) + (pixels_height / screen_height_in_mm)) / 2

    # adjust time tag to start at 0
    eye_data["time_tag"] = eye_data.TimeTag - eye_data.TimeTag[0]

    # annotate binocular fixations
    eye_data["Fixation"] = eye_data.LeftEyeFixationFlag + eye_data.RightEyeFixationFlag
    ## eliminate simultaneous blink and fixation (setting fixation to 0)
    eye_data.Fixation.loc[eye_data.LeftBlink > 0.0] = 0.0
    eye_data.Fixation.loc[eye_data.RightBlink > 0.0] = 0.0
    eye_data.Fixation[eye_data.Fixation > 1] = 1.0

    # annotate binocular saccades
    eye_data["Saccade"] = eye_data.LeftEyeSaccadeFlag + eye_data.RightEyeSaccadeFlag
    ## eliminate simultaneous blink and saccades (setting saccade to 0)
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
    eye_data["saccade_direction_x"] = np.nan
    eye_data["saccade_direction_y"] = np.nan
    eye_data["saccade_amplitude"] = np.nan
    out = eye_data.groupby("N_saccade", dropna=False, group_keys=False).apply(calc_saccade_direction)

    # convert saccade amplitude from pixels to visual angle (°)
    # eye_data["saccade_amplitude_visual_angle"] =

    # set saccade direction to NaN everywhere where there is no saccade
    out.loc[eye_data.Saccade < 1.0, ["saccade_direction_x", "saccade_direction_y"]] = np.nan

    # sum up left and right eye positions to converging eye position in x and y dimension
    out["converging_eye_x"] = out.apply(lambda row: (row.LeftEyeX + row.RightEyeX) / 2, axis=1)
    out["converging_eye_y"] = out.apply(lambda row: (row.LeftEyeY + row.RightEyeY) / 2, axis=1)

    return out


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
