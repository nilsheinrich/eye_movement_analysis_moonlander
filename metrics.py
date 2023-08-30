import numpy as np
import pandas as pd
from helper_functions import pixel_to_degree


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

    timeTag = eye_data.time_tag
    eyeX = eye_data.converging_eye_x_adjusted
    eyeY = eye_data.converging_eye_y_adjusted
    exploring = eye_data.exploring_fixation

    if exploring == 0:
        # get row from input_data_ with closest match in time
        input_row = input_data.iloc[(input_data['time_played'] - timeTag).abs().argsort()[:1]]

        # get player position n pixel
        playerPosX = input_row.player_pos.values[0][0]
        playerPosY = input_row.player_pos.values[0][1]

        # compute distance to spaceship for each regressive saccade
        dists = pixel_to_degree(np.sqrt(np.power(playerPosX - eyeX, 2) + np.power(playerPosY - eyeY, 2)))

    return dists


def get_dist_to_spaceship_fix_rest_multiple(eye_data, input_data):
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
        dist = pixel_to_degree(np.sqrt(np.power(playerPosX - row.eyeX, 2) + np.power(playerPosY - row.eyeY, 2)))
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

    timeTag = eye_data.time_tag
    eyeX = eye_data.converging_eye_x_adjusted
    eyeY = eye_data.converging_eye_y_adjusted
    exploring = eye_data.exploring_fixation

    if exploring == 1:
        # get row from input_data_ with closest match in time
        input_row = input_data.iloc[(input_data['time_played'] - timeTag).abs().argsort()[:1]]

        # get player position n pixel
        playerPosX = input_row.player_pos.values[0][0]
        playerPosY = input_row.player_pos.values[0][1]

        # compute distance to spaceship for each regressive saccade
        dists = pixel_to_degree(np.sqrt(np.power(playerPosX - eyeX, 2) + np.power(playerPosY - eyeY, 2)))

    return dists


def get_dist_to_obstacles_fix_explore_multiple(eye_data, input_data):
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

    timeTag = eye_data.time_tag
    eyeX = eye_data.converging_eye_x_adjusted
    eyeY = eye_data.converging_eye_y_adjusted
    saccDirX = eye_data.saccade_direction_x
    saccDirY = eye_data.saccade_direction_y

    # target saccades
    if (target_saccades == 'progress' and saccDirY < 0) or (target_saccades == 'regress' and saccDirY > 0):
        # get row from input_data_ with closest match in time
        input_row = input_data.iloc[(input_data['time_played'] - timeTag).abs().argsort()[:1]]

        # get player position n pixel
        saccadeLandX = eyeX + saccDirX
        saccadeLandY = eyeY + saccDirY

        # compute distance to spaceship for each regressive saccade
        obsDists = []
        for obstacle in input_row.visible_obstacles.values[0]:
            dist = np.sqrt(np.power(obstacle[0] - saccadeLandX, 2) + np.power(obstacle[1] - saccadeLandY, 2))
            obsDists.append(pixel_to_degree(dist))
        # the shortest distance is the obstacle most likely in focus of visual attention
        if len(obsDists) > 0:
            dists.append(min(obsDists))

    return dists


def get_dist_to_obstacles_sacc_multiple(eye_data, input_data, target_saccades='regress'):
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
        temp_df_targets = temp_df[temp_df.regressiveSaccade is True]
    elif target_saccades == 'progress':
        temp_df_targets = temp_df[temp_df.regressiveSaccade is False]

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
