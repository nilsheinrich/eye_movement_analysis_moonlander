import pandas as pd
import re
from ast import literal_eval
from plotting_functions import situational_analysis

crash_success_runs = pd.read_csv("crash_runs/crash_success_runs.csv", index_col=False)

# metrics we are interest in their difference from crash and success
metric_columns = ['fixDurs_crash',
                  'fixLocsY_crash',
                  'saccAmps_crash',
                  'distSpaceship_fix_rest_crash',
                  'distClosestObstacle_sacc_progress_crash',
                  'distClosestObstacle_sacc_regress_crash',
                  'distClosestObstacle_fix_explore_crash',
                  'fixDurs_success',
                  'fixLocsY_success',
                  'saccAmps_success',
                  'distSpaceship_fix_rest_success',
                  'distClosestObstacle_sacc_progress_success',
                  'distClosestObstacle_sacc_regress_success',
                  'distClosestObstacle_fix_explore_success']


for row in range(len(crash_success_runs[:2])):

    for col in metric_columns:
        cell = crash_success_runs.iloc[row][col]

        cell = re.sub(' +', ' ', cell)

        cell = cell.replace('nan,', '').replace('nan', '')
        cell = cell.replace('[ ', '[').replace(' ]', ']')
        cell = cell.replace(' ', ',').replace('\n', '')

        cell = literal_eval(cell)

        crash_success_runs.at[row, f'{col}'] = cell

situational_analysis(data=crash_success_runs[:2], safe_plot=False, debug=False)
