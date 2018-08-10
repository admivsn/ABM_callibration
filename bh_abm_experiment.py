from bh_abm import *
from generic_abm import *

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# Set the ABM Evaluation Budget
n_train_samples = 100

# Set test size
n_test_samples = 100

# Set the ABM parameters and support
bh_abm_exploration_range = np.array([(0.0, 10.0),  # beta  (intensity of choice)
                                     (0.5, 0.5),   # n_1   (initial share of type 1 traders)
                                     (-2.0, 2.0),  # b_1   (bias of type 1 traders)
                                     (-2.0, 2.0),  # b_2   (bias of type 2 traders)
                                     (-2.0, 2.0),  # g_1   (trend of type 1 traders)
                                     (-2.0, 2.0),  # g_2   (trend of type 2 traders)
                                     (0.0, 5.0),   # C     (cost of obtaining type 1 forecasts)
                                     (0.0, 1.0),   # w     (weight to past profits)
                                     (0.0, 1.0),   # sigma (asset volatility)
                                     (0.0, 100.0), # v     (attitude towards risk)
                                     (0.01, 0.1)]) # r    (risk free return)

# Get parameters for train and test from the support
X_train, X_test = get_unirand_parameter_samples(n_train_samples, n_test_samples, bh_abm_exploration_range)

# Run the ABM on the train and test parameters
y_train, y_test =  bh_abm_evaluate_samples(X_train, X_test)
