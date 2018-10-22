from bh_abm import *
from generic_abm import *

import pandas as pd

def bh_abm_simulator(n_samples, callibration):

    # Set the ABM parameters and support
    bh_abm_exploration_range = np.array([(0.0, 10.0),  # beta  (intensity of choice)
                                         (0.5, 0.5),   # n_1   (initial share of type 1 traders) Kept constant at 0.5
                                         (-2.0, 2.0),  # b_1   (bias of type 1 traders)
                                         (-2.0, 2.0),  # b_2   (bias of type 2 traders)
                                         (-2.0, 2.0),  # g_1   (trend of type 1 traders)
                                         (-2.0, 2.0),  # g_2   (trend of type 2 traders)
                                         (0.0, 5.0),   # C     (cost of obtaining type 1 forecasts)
                                         (0.0, 1.0),   # w     (weight to past profits)
                                         (0.0, 1.0),   # sigma (asset volatility)
                                         (0.0, 100.0), # v     (attitude towards risk)
                                         (0.01, 0.1)]) # r    (risk free return)    
    
    X, ignore = get_unirand_parameter_samples(n_samples, 1, bh_abm_exploration_range)
    
    if callibration == "real_valued":
    
        # Run the ABM on the train and test parameters in REALVALUED case...
        y, ignore =  bh_abm_evaluate_samples(X, ignore, "real_valued")
    
    elif callibration == "binary":
    
        # Run the ABM on the train and test parameters in the BINARY case
        y, ignore =  bh_abm_evaluate_samples(X, ignore, "binary")
        
    pd.DataFrame(X).to_csv("bh_samples_X_real_valued.csv")
    pd.DataFrame(y).to_csv("bh_samples_y_real_valued.csv")
    
