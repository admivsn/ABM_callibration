from island_abm import *
from generic_abm import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt





# Set the ABM Evaluation Budget
n_train_samples = 100

# Set test size
n_test_samples = 100

# Set the ABM parameters and support
island_abm_exploration_range = np.array([(0.0, 10.0),  # rho            (degree of locality in the diffustion of knowledge)
                                         (0.0, 5.0),   # lambda_param   (mean of Poisson r.v. - jumps in technology)
                                         (0.8, 2.0),   # alpha          (productivity of labour in extraction)
                                         (0.0, 1.0),   # phi            (cumulative learning effect)
                                         (0.0, 1.0),   # pi             (probability of finding a new island)
                                         (0.0, 1.0)])  # eps           (willingness to explore)

# Get parameters for train and test from the support
X_train, X_test = get_unirand_parameter_samples(n_train_samples, n_test_samples, island_abm_exploration_range)

#####

# Run the ABM on the train and test parameters in BINARY case...
y_train, y_test =  island_abm_evaluate_samples(X_train, X_test, "binary")

# Make dataframe of the training data
df_binary = pd.concat([pd.DataFrame(X_train, columns = ["rho", "lambda_param", "alpha", "phi", "pi", "eps"]), 
                       pd.DataFrame(y_train, columns = ["result"])], 
                       axis = 1)

# Plot the prior distributions of each parameter
island_abm_make_plot(df_binary, 
                     "(Binary) Prior distributions for " + str(df_binary.shape[0]) + " simulations")

# Subset the dataframe to just the accepted parameters
df_binary_success = df_binary[df_binary["result"] == 1.0]

# Plot the posterior distributions of each parameter
island_abm_make_plot(df_binary_success, 
                     "(Binary) Posterior distributions for " + str(df_binary_success.shape[0]) + " acceptances")

#####

# Run the ABM on the train and test parameters in REALVALUED case...
y_train, y_test =  island_abm_evaluate_samples(X_train, X_test, "real_valued")

# Make dataframe of the training data
df_realvalued = pd.concat([pd.DataFrame(X_train, columns = ["rho", "lambda_param", "alpha", "phi", "pi", "eps"]), 
                           pd.DataFrame(y_train, columns = ["result"])], 
                           axis = 1)

# Plot the prior distributions of each parameter
island_abm_make_plot(df_realvalued, 
                     "(Real Valued) Prior distributions for " + str(df_realvalued.shape[0]) + " simulations")

# Subset the dataframe to just the accepted parameters, this is where we define eps
eps = 1
df_realvalued_success = df_realvalued[df_realvalued["result"] > eps]
 
# Plot the posterior distributions of each parameter
island_abm_make_plot(df_realvalued_success, 
                     "(Real Valued) Posterior distributions for " + str(df_realvalued_success.shape[0]) + " acceptances")

#####