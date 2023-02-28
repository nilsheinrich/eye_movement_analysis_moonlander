import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from helper_functions import pre_process_input_data, pre_process_eye_data, point_estimate
import matplotlib.animation as animation
# bad practice but let's suppress warnings
import warnings

warnings.filterwarnings("ignore")


def visualize_sequence(id_code, done_, n_run, arg_comb, start_time, end_time, safe_ani=False):
    """
    :param id_code: experimental code of participant
    :param done_: done vs. crashed - used to identify input_data file
    :param n_run: experimental trial
    :param arg_comb: triplet of level, drift, and input_noise
    :param start_time: in s - begin render_gaming_sequence here
    :param end_time: in s - stop render_gaming_sequence here
    :param safe_ani: True vs False - whether to export sequence to mp4
    """
    # load and pre_process data by given code and n_run
    # (with input data being button presses and eye data being eye tracking output)
    input_data = pd.read_csv(f'experimental_data/{id_code}/data/{id_code}_output_{arg_comb}_{done_}_{n_run:0>2}.csv', index_col=False)
    input_data = pre_process_input_data(input_data)

    eye_data = pd.read_csv(f'experimental_data/{id_code}/eye_data/{id_code}_eye_tracking_output_{arg_comb}_{n_run:0>2}.csv', index_col=False)
    eye_data = pre_process_eye_data(eye_data)

    # call visualize function
    render_gaming_sequence(input_data=input_data, eye_data=eye_data, start_time=start_time, end_time=end_time, safe_ani=safe_ani)


def render_gaming_sequence(input_data, eye_data, start_time, end_time, scaling=18, edge=34, observation_space_x=40,
                           observation_space_y=60, bottom_edge=15, obstacle_size=2, drift_size=15, agent_size_x=2,
                           agent_size_y=2, safe_ani=False):
    """
    :param input_data: pandas DataFrame; data generated by moonlander experiment. Contains button presses and visible objects
    :param eye_data: pandas DataFrame; data generated by eye tracker of any frequency. Matching of input_ and eye_data is done within this function.
    :param start_time: point in time to begin visualization
    :param end_time: point in time until when sequence shall be visualized
    :param safe_ani: True vs False - whether to export sequence to mp4
    the other passed variables are parameters for visualization determined in config.py and should be fixed until experimental setup changes.
    """

    # adjust visualization parameters by scaling
    edge = edge * scaling
    observation_space_x = observation_space_x * scaling
    observation_space_y = (observation_space_y - bottom_edge) * scaling

    obstacle_size = obstacle_size * scaling
    drift_size = drift_size * scaling
    agent_size_x, agent_size_y = agent_size_x * scaling, agent_size_y * scaling

    # subset data by given time interval
    input_data_ = input_data[input_data.time_played.between(start_time, end_time)]
    eye_data_ = eye_data[eye_data.time_tag.between(start_time, end_time)]
    factor = math.floor(len(eye_data_) / len(input_data_))

    # eliminate nans by replacing with preceeding value
    eye_data_.converging_eye_x_adjusted.fillna(method='ffill', inplace=True)
    eye_data_.converging_eye_y_adjusted.fillna(method='ffill', inplace=True)

    fig, ax = plt.subplots(figsize=(8, 9))

    obstacles, = ax.plot([], [], color='grey', marker='o', fillstyle='full', markersize=obstacle_size, alpha=0.8)

    def animate_frame(i):
        # clear axe and reset axes limits
        plt.cla()
        ax.set_xlim((edge, observation_space_x + edge))
        ax.set_ylim(observation_space_y)
        # hide axes
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())

        # input keys
        current_input = input_data_.current_input.iloc[i]  # nan vs. "Left" vs. "Right"
        # assign alpha based on input
        alpha_left = 0.2
        alpha_right = 0.2
        if current_input == "Left":  # Left -> right and vice versa due to inverse environmental movement around agent
            alpha_right += 0.8
        if current_input == "Right":
            alpha_left += 0.8

        # draw arrows for visualizing current input
        triangle_points_right = [[620, 60], [700, 10], [700, 110]]
        key_right = plt.Polygon(triangle_points_right, color='lightgreen', alpha=alpha_right)
        ax.add_patch(key_right)
        triangle_points_left = [[800, 60], [720, 10], [720, 110]]
        key_left = plt.Polygon(triangle_points_left, color='lightgreen', alpha=alpha_left)
        ax.add_patch(key_left)

        # obstacles
        obstacle_instance = input_data_.visible_obstacles.iloc[i]
        obstacles_data = pd.DataFrame(obstacle_instance, columns=['x', 'y'])

        drift_instance = input_data_.visible_drift_tiles.iloc[i]
        drift_data = pd.DataFrame(drift_instance, columns=['x', 'y'])
        # transform x to just be boarder of plot (left = edge vs. right = observation_space_x + edge
        drift_data['x'] = drift_data['x'].apply(lambda x: ax.get_xlim()[0] if x <= edge else ax.get_xlim()[1]-10)

        # draw each obstacle on canvas individually
        for index, obstacle in obstacles_data.iterrows():
            ax.plot(obstacle.x + obstacle_size / 2, obstacle.y + obstacle_size / 2, color='grey', marker='o',
                    fillstyle='full', markersize=obstacle_size, alpha=0.8)

        # draw each drift tile on canvas individually
        for index, drift_tile in drift_data.iterrows():
            drift_rect = plt.Rectangle((drift_tile.x, drift_tile.y), color='red', width=10, height=drift_size, alpha=0.3)
            ax.add_patch(drift_rect)

        # eye tracking data
        # subset for given point in time of input data
        eye_data_subset = eye_data_[i * factor: i * factor + factor]
        sns.kdeplot(x=eye_data_subset.converging_eye_x_adjusted,
                    y=eye_data_subset.converging_eye_y_adjusted,
                    cmap="Reds",
                    shade=True,
                    alpha=0.9,
                    ax=ax)
        # draw fixation
        y_coord = point_estimate(eye_data_subset.converging_eye_y_adjusted)[0]
        ax.axhline(y_coord, color="crimson")
        x_coord = point_estimate(eye_data_subset.converging_eye_x_adjusted)[0]
        ax.axvline(x_coord, color="crimson")

        # agent
        # create array of 3 points (triangle) for drawing agent
        player_position = input_data_.player_pos.iloc[0]
        triangle_points = [player_position,
                           [player_position[0] + agent_size_x, player_position[1]],
                           [player_position[0] + (agent_size_x / 2), player_position[1] + agent_size_y]
                           ]
        agent = plt.Polygon(triangle_points, color='lightgreen')
        ax.add_patch(agent)

        return obstacles,

    ani = animation.FuncAnimation(fig=plt.gcf(), func=animate_frame, frames=len(input_data_), interval=100, blit=True, repeat=False)

    if safe_ani:
        FFwriter = animation.FFMpegWriter(fps=10)
        ani.save('videos/animation.mp4', writer=FFwriter)

    plt.show()
