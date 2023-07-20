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

    if 'Unnamed: 0' in dataframe.columns:
        dataframe.rename(columns={'Unnamed: 0': 'frame'}, inplace=True)
    else:
        dataframe.insert(0, 'frame', dataframe.index)

    # convert columns with literal_eval to not have strings anymore
    dataframe.player_pos = dataframe.player_pos.apply(lambda row: literal_eval(row))
    dataframe.visible_obstacles = dataframe.visible_obstacles.apply(lambda row: literal_eval(row))
    dataframe.visible_drift_tiles = dataframe.visible_drift_tiles.apply(lambda row: literal_eval(row))
    if 'last_walls_tile' in dataframe:
        dataframe.last_walls_tile = dataframe.last_walls_tile.apply(lambda row: literal_eval(row))

    # adjust time tag if existent
    if 'time_tag' in dataframe.columns:
        dataframe['adjusted_time_tag'] = dataframe.time_played + dataframe.time_tag

    ## annotate input data

    # input can be either None, "Right", "Left"
    rows_with_input = dataframe[~dataframe["current_input"].isnull()][["frame"]]

    # condition for start input
    cond = rows_with_input.frame - rows_with_input.frame.shift(1) >= 2

    # have =1 everywhere condition applies and =0 where not
    rows_with_input["start_input"] = np.where(cond, 1, 0)

    # flag first row also as start input (because we missed it by not taking any rows before that row due to subsetting)
    # but account for data without any input
    if len(rows_with_input) > 0:
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

    # include input_duration for every button press
    try:
        dataframe['input_duration'] = np.nan
        dataframe['input_duration'] = dataframe.groupby('N_input')['time_played'].transform(lambda x: max(x) - min(x))
    except ValueError:
        print("ValueError: Length mismatch: Expected axis has X elements, new values have Y elements")

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


def calc_fixation_duration(fixation_rows):
    """
    Need to be given pandas dataframe grouped by N_fixation.
    Dataframe must have columns of TimeTag (VPixx generated data).
    Returned dataframe now holds fixation duration.
    """

    fixation_rows.fixation_duration = fixation_rows.iloc[-1].TimeTag - fixation_rows.iloc[0].TimeTag

    return fixation_rows


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
    saccade_rows.saccade_amplitude_in_pixel = saccade_rows.apply(
        lambda x: np.sqrt(np.power(x_direction, 2) + np.power(y_direction, 2)), axis=1)

    return saccade_rows


def pixel_to_degree(distance_on_screen_pixel, mm_per_pixel=595 / 1920, distance_to_screen_mm=840):
    """
    calculate the visual degrees of a distance (saccade amplitude, on screen distance between objects)
    setup:
     - screen_width_in_mm=595
     - screen_height_in_mm=335
     - pixels_screen_width=1920
     - pixels_screen_height=1080
     - distance_to_screen_in_mm=840 ; Nikita measured 770mm
    """
    distance_on_screen_mm = float(distance_on_screen_pixel) * mm_per_pixel

    visual_angle_in_radians = np.arctan(distance_on_screen_mm / distance_to_screen_mm)

    return np.rad2deg(visual_angle_in_radians)


# annotate eye_tracking data

def pre_process_eye_data(eye_data, spaceship_center_x=972, spaceship_center_y=288):
    """
    dataframe must be pandas dataFrame with appropriate columns...
    """

    # adjust time tag to start at 0
    eye_data["time_tag"] = eye_data.TimeTag - eye_data.TimeTag[0]

    # annotate binocular fixations
    eye_data["Fixation"] = eye_data.LeftEyeFixationFlag + eye_data.RightEyeFixationFlag
    ## eliminate simultaneous blink and fixation (setting fixation to 0)
    eye_data.Fixation.loc[eye_data.LeftBlink > 0.0] = 0.0
    eye_data.Fixation.loc[eye_data.RightBlink > 0.0] = 0.0
    eye_data.Fixation[eye_data.Fixation > 1] = 1.0

    # condition for initiating fixation
    cond = (eye_data.Fixation >= 1.0) & (eye_data.Fixation.shift(1) == 0.0)

    # have =1 everywhere condition applies and =0 where not
    eye_data["fixationOnset"] = np.where(cond, 1, 0)

    # insert N_fixation - counting up fixations
    eye_data["N_fixation"] = (eye_data["fixationOnset"] == 1).cumsum()
    eye_data.loc[eye_data.Fixation < 1.0, "N_fixation"] = np.nan  # have NaN everywhere where there is no fixation

    # annotate fixation duration
    eye_data["fixation_duration"] = np.nan
    eye_data = eye_data.groupby("N_fixation", dropna=False).apply(calc_fixation_duration)

    # sum up left and right eye positions to converging eye position in x and y dimension
    eye_data["converging_eye_x"] = eye_data.apply(lambda row: (row.LeftEyeX + row.RightEyeX) / 2, axis=1)
    eye_data["converging_eye_y"] = eye_data.apply(lambda row: (row.LeftEyeY + row.RightEyeY) / 2, axis=1)

    # adjust eye-tracking coordinates by fixed factor
    eye_data["converging_eye_x_adjusted"] = eye_data.converging_eye_x + 960
    eye_data["converging_eye_y_adjusted"] = eye_data.converging_eye_y.apply(lambda x: x * (-1) + 540)

    # annotate distance to spaceship
    eye_data["distance_to_spaceship_in_pixel"] = np.sqrt(
        np.power((eye_data.converging_eye_x_adjusted - spaceship_center_x), 2) +
        np.power((eye_data.converging_eye_y_adjusted - spaceship_center_y), 2))
    eye_data["distance_to_spaceship"] = eye_data['distance_to_spaceship_in_pixel'].apply(lambda x: pixel_to_degree(x))

    # annotate fixations exploring the scene (further than 4° visual angle from spaceship - outside of parafovea)
    cond = (eye_data["fixationOnset"] == 1.0) & (eye_data["distance_to_spaceship"] > 4)
    # have =1 everywhere condition applies and =0 where not
    eye_data["exploring_fixation"] = np.where(cond, 1, 0)

    # flag fixations and saccades aiming within game boarders
    # in (edge*scaling, (edge+observation_space_x)*scaling)

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
    eye_data["saccade_amplitude_in_pixel"] = np.nan
    out = eye_data.groupby("N_saccade", dropna=False).apply(calc_saccade_direction)

    # convert saccade amplitude from pixels to visual angle (°)
    out["saccade_amplitude"] = out.saccade_amplitude_in_pixel.apply(lambda x: pixel_to_degree(x))

    # set saccade direction to NaN everywhere where there is no saccade
    out.loc[eye_data.Saccade < 1.0, ["saccade_direction_x", "saccade_direction_y"]] = np.nan

    return out


def point_estimate(data, hpdi=0.25, n_samples=1000):
    """
    function for estimating point of maximum density for passed data
    :param data: data for which a KDE is instantiated and point estimates are derived
    :param hpdi: highest probability density interval - boarders (upper and lower) will be returned where hpdi is
    derived for samples
    :param n_samples: number of samples which are sampled from KDE
    """
    try:
        kde = st.gaussian_kde(data)  # gaussian kernel
        samples = kde.resample(n_samples)[0]  # sampling
        steps = np.linspace(min(data), max(data), n_samples)  # building space
        probs = kde.evaluate(steps)  # get likelihood for every step in space
        point_estimate_y = max(probs)  # get highest likelihood value
        point_estimate_index = probs.argmax()
        point_estimate_x = steps[point_estimate_index]

        # compute hpdi - computing boarders for the smallest interval which contains hpdi(%) of the mass
        hdi = az.hdi(samples, hdi_prob=hpdi)
        return point_estimate_x, point_estimate_y, hdi[0], hdi[1], samples
    except np.linalg.LinAlgError:
        print("SingularMatrixError; numpy.linalg.LinAlgError: singular matrix; no variance in data")

        return data.iloc[0], data.iloc[0], data.iloc[0], data.iloc[0], data.iloc[0]


def get_dist_to_spaceship_fix_rest(eye_data, input_data):
    """
    For each resting fixation the appropriate row in input_data is checked for the position of the spaceship.
    Computes the euclidean distance of converging point of eyes and spaceship (sqrt of squared x and y).
    Returns array of distances.

    eye_data and input_data must be already subsetted to the desired time period. If you want the distances
    over the whole level, simply pass the complete eye_ and input_data.
    """
    # empty list to-be-returned
    dists = []

    timeTags = eye_data[eye_data.fixationOnset == 1].time_tag

    eyeX = eye_data[eye_data.fixationOnset == 1].converging_eye_x_adjusted
    eyeY = eye_data[eye_data.fixationOnset == 1].converging_eye_y_adjusted

    exploring = eye_data[eye_data.fixationOnset == 1].exploring_fixation

    # putting together temp_df
    temp = {'timeTag': timeTags, 'eyeX': eyeX, 'eyeY': eyeY, 'exploring_fixation': exploring}
    temp_df = pd.DataFrame(data=temp)

    # only resting fixations are of interest for dist to spaceship
    temp_df_rest = temp_df[temp_df["exploring_fixation"] == 0]

    for index, row in temp_df_rest.iterrows():
        # get row from input_data_ with closest match in time
        input_row = input_data.iloc[(input_data['time_played'] - row.timeTag).abs().argsort()[:1]]

        # get player position n pixel
        playerPosX = input_row.player_pos.values[0][0]
        playerPosY = input_row.player_pos.values[0][1]

        # compute distance to spaceship for each regressive saccade
        dist = np.sqrt(np.power(playerPosX - row.eyeX, 2) + np.power(playerPosY - row.eyeY, 2))
        dists.append(pixel_to_degree(dist))

    return dists


def get_dist_to_obstacles_fix_explore(eye_data, input_data):
    """
    For each exploring fixation the appropriate row in input_data is checked for the all obstacles on screen
    and their positions.
    Computes the euclidean distance of converging point of eyes and the closest obstacle (sqrt of squared x and y).
    Returns array of distances.

    eye_data and input_data must be already subsetted to the desired time period. If you want the distances
    over the whole level, simply pass the complete eye_ and input_data.
    """
    # empty list to-be-returned
    dists = []

    timeTags = eye_data[eye_data.fixationOnset == 1].time_tag

    eyeX = eye_data[eye_data.fixationOnset == 1].converging_eye_x_adjusted
    eyeY = eye_data[eye_data.fixationOnset == 1].converging_eye_y_adjusted

    exploring = eye_data[eye_data.fixationOnset == 1].exploring_fixation

    # putting together temp_df
    temp = {'timeTag': timeTags, 'eyeX': eyeX, 'eyeY': eyeY, 'exploring_fixation': exploring}
    temp_df = pd.DataFrame(data=temp)

    # only exploring fixations are of interest for dist to obstacles
    temp_df_explore = temp_df[temp_df["exploring_fixation"] == 1]

    for index, row in temp_df_explore.iterrows():
        # get row from input_data_ with closest match in time
        input_row = input_data.iloc[(input_data['time_played'] - row.timeTag).abs().argsort()[:1]]

        # compute distances of all obstacles on screen to saccade landing site
        obsDists = []
        for obstacle in input_row.visible_obstacles.values[0]:
            dist = np.sqrt(np.power(obstacle[0] - row.eyeX, 2) + np.power(obstacle[1] - row.eyeY, 2))
            obsDists.append(pixel_to_degree(dist))
        # the shortest distance is the obstacle most likely in focus of visual attention
        if len(obsDists) > 0:
            dists.append(min(obsDists))

    return dists


def get_dist_to_obstacles_sacc(eye_data, input_data, target_saccades='regress'):
    """
    :param target_saccades: can either be 'regress' (default) or 'progress'.
    The parameter target saccades determines which type of saccades are considered: regress for upwards saccades and
    progress for downwards saccades.
    For each target saccade the appropriate row in input_data is checked for the all obstacles on screen
    and their positions.
    Computes the euclidean distance of the saccade landing site and the closest obstacle (sqrt of squared x and y).
    Returns array of distances.

    eye_data and input_data must be already subsetted to the desired time period. If you want the distances
    over the whole level, simply pass the complete eye_ and input_data.
    """
    # empty list to-be-returned
    dists = []

    timeTags = eye_data[eye_data.saccadeOnset == 1].time_tag

    eyeX = eye_data[eye_data.saccadeOnset == 1].converging_eye_x_adjusted
    eyeY = eye_data[eye_data.saccadeOnset == 1].converging_eye_y_adjusted

    saccDirX = eye_data[eye_data.saccadeOnset == 1].saccade_direction_x
    saccDirY = eye_data[eye_data.saccadeOnset == 1].saccade_direction_y

    # putting together temp_df
    temp = {'timeTag': timeTags, 'eyeX': eyeX, 'eyeY': eyeY, 'saccDirX': saccDirX, 'saccDirY': saccDirY}
    temp_df = pd.DataFrame(data=temp)

    # flagging regressive saccades
    temp_df['regressiveSaccade'] = np.where(temp_df['saccDirY'] < 0, True, False)

    temp_df['saccadeLandX'] = temp_df.eyeX + temp_df.saccDirX
    temp_df['saccadeLandY'] = temp_df.eyeY + temp_df.saccDirY

    # filtering for target saccades
    if target_saccades == 'regress':
        temp_df_targets = temp_df[temp_df.regressiveSaccade == True]
    elif target_saccades == 'progress':
        temp_df_targets = temp_df[temp_df.regressiveSaccade == False]

    for index, row in temp_df_targets.iterrows():
        # get row from input_data_ with closest match in time
        input_row = input_data.iloc[(input_data['time_played'] - row.timeTag).abs().argsort()[:1]]

        # compute distances of all obstacles on screen to saccade landing site
        obsDists = []
        for obstacle in input_row.visible_obstacles.values[0]:
            dist = np.sqrt(np.power(obstacle[0] - row.saccadeLandX, 2) + np.power(obstacle[1] - row.saccadeLandY, 2))
            obsDists.append(pixel_to_degree(dist))
        # the shortest distance is the obstacle most likely in focus of visual attention
        if len(obsDists) > 0:
            dists.append(min(obsDists))

    return dists
