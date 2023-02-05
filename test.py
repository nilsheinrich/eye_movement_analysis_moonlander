import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from helper_functions import pre_process_input_data, pre_process_eye_data, point_estimate
from matplotlib.animation import FuncAnimation
# bad practice but let's suppress warnings
import warnings
warnings.filterwarnings("ignore")


# visualization parameters
scaling = 18
edge = 34*scaling

observation_space_x = 40*scaling

bottom_edge = 15
observation_space_y = (60 - bottom_edge)*scaling

obstacle_size = 2*scaling
agent_size_x, agent_size_y = 2*scaling, 2*scaling

# drift_range = 15*scaling

# data
code = "pilot4"
n_run = 11

# load data (with input data being button presses and eye data being eye tracking output)
input_data = pd.read_csv(f'input_data/{code}_output_{n_run:0>2}.csv', index_col=False)
input_data = pre_process_input_data(input_data)

eye_data = pd.read_csv(f'eye_data/{code}_eye_tracking_output_{n_run:0>2}.csv', index_col=False)
eye_data = pre_process_eye_data(eye_data)

# time interval
start_time = 28
end_time = 45

# subset data
input_data_ = input_data[input_data.time_played.between(start_time, end_time)]
eye_data_ = eye_data[eye_data.time_tag.between(start_time, end_time)]

factor = math.floor(len(eye_data_) / len(input_data_))

# adjust eye-tracking coordinates by fixed factor
eye_data_["converging_eye_x_adjusted"] = eye_data_.converging_eye_x + 960
eye_data_["converging_eye_y_adjusted"] = eye_data_.converging_eye_y.apply(lambda x: x*(-1)+540)

# eliminate nans by replacing with preceeding value
eye_data_.converging_eye_x_adjusted.fillna(method='ffill', inplace=True)
eye_data_.converging_eye_y_adjusted.fillna(method='ffill', inplace=True)


fig, ax = plt.subplots(figsize=(8, 9))

obstacles, = ax.plot([], [], color='grey', marker='o', fillstyle='full', markersize=obstacle_size, alpha=0.8)


# def init():
#     ax.set_xlim((edge, observation_space_x + edge))
#     ax.set_ylim(observation_space_y)
#     return obstacles,


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

    # draw arrows
    triangle_points_right = [[620, 60], [700, 10], [700, 110]]
    key_right = plt.Polygon(triangle_points_right, color='lightgreen', alpha=alpha_right)
    ax.add_patch(key_right)
    triangle_points_left = [[800, 60], [720, 10], [720, 110]]
    key_left = plt.Polygon(triangle_points_left, color='lightgreen', alpha=alpha_left)
    ax.add_patch(key_left)

    # obstacles
    instance = input_data_.visible_obstacles.iloc[i]
    obstacles_data = pd.DataFrame(instance, columns=['x', 'y'])

    # draw each obstacle on canvas individually
    for index, obstacle in obstacles_data.iterrows():
        ax.plot(obstacle.x + obstacle_size / 2, obstacle.y + obstacle_size / 2, color='grey', marker='o',
                fillstyle='full', markersize=obstacle_size, alpha=0.8)

    # eye tracking data
    # subset
    eye_data_subset = eye_data_[i*factor: i*factor + factor]
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


ani = FuncAnimation(fig=plt.gcf(), func=animate_frame, frames=len(input_data_), interval=100, blit=True)

plt.show()
