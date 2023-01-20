import numpy as np
from helper_functions import plot_kde_combined


N_run = 1

for run in np.arange(N_run):
    plot_kde_combined(code="pilot4", n_run=run, safe_plot=False)
