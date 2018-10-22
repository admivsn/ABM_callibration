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

# Define Monte Carlo simulations
n_monte_carlo = 2

# Final test set size
n_final_test_samples = n_test_samples * n_monte_carlo

# Set the ABM parameters and support
island_abm_exploration_range = np.array([(0.0, 10.0),  # rho            (degree of locality in the diffustion of knowledge)
                                         (0.0, 5.0),   # lambda_param   (mean of Poisson r.v. - jumps in technology)
                                         (0.8, 2.0),   # alpha          (productivity of labour in extraction)
                                         (0.0, 1.0),   # phi            (cumulative learning effect)
                                         (0.0, 1.0),   # pi             (probability of finding a new island)
                                         (0.0, 1.0)])  # eps           (willingness to explore)

# Get parameters for train and test from the support
X_train, X_test = get_unirand_parameter_samples(n_train_samples, n_test_samples, island_abm_exploration_range)

# Run the ABM on the train and test parameters in  REALVALUED case...
y_train, y_test =  island_abm_evaluate_samples(X_train, X_test, "real_valued")

# Train a LinearRegression model
LinearRegression_model = fit_surrogate_as_linear_regression(X_train, y_train)

# Train a surrogate GaussianProcessRegressor model
GaussianProcessRegressor_model = fit_surrogate_as_gp_regression(X_train, y_train)

# Train a surrogate GradientBoostingRegressor model
GradientBoostingRegressor_model = fit_surrogate_as_gbt_regression(X_train, y_train)

# Evaluate the surrogates on the test set
y_hat_test = [None] * 3
y_hat_test[0] = LinearRegression_model.predict(X_test)
y_hat_test[1] = GaussianProcessRegressor_model.predict(X_test)
y_hat_test[2] = GradientBoostingRegressor_model.predict(X_test)

# MSE performance
mse_performance = np.zeros((3, n_monte_carlo))
for sur_idx in range(len(y_hat_test)):
    for i in range(n_monte_carlo):
        print "sur_idx", sur_idx, "i", i
        mse_performance[sur_idx, i] = mean_squared_error(y_test[i * n_test_samples:(i + 1) * n_test_samples], y_hat_test[int(sur_idx)][i * n_test_samples:(i + 1) * n_test_samples])
         
# Plot the Monte Carlo performance densities for each of the methods
experiment_labels = ["LinearRegression", "GaussianProcessRegressor", "GradientBoostingRegressor"]

mse_performance = pd.DataFrame(mse_performance, index = experiment_labels)

LinearRegression_label = "LinearRegression: Mean " + '{:2.5f}'.format(mse_performance.iloc[0, :].mean()) + ", Variance " + '{:2.5f}'.format(mse_performance.iloc[0, :].var())
GaussianProcessRegressor_label = "GaussianProcessRegressor: Mean " + '{:2.5f}'.format(mse_performance.iloc[1, :].mean()) + ", Variance " + '{:2.5f}'.format(mse_performance.iloc[1, :].var())
GradientBoostingRegressor_label = "GradientBoostingRegressor: Mean " + '{:2.5f}'.format(mse_performance.iloc[2, :].mean()) + ", Variance " + '{:2.5f}'.format(mse_performance.iloc[2, :].var())

fig, ax = plt.subplots(figsize=(12, 5))
sns.distplot(mse_performance.iloc[0, :], label = LinearRegression_label, ax = ax)
sns.distplot(mse_performance.iloc[1, :], label = GaussianProcessRegressor_label, ax = ax)
sns.distplot(mse_performance.iloc[2, :], label = GradientBoostingRegressor_label, ax = ax)

plt.title("Prediction Performance")
plt.xlabel('Mean-Squared Error')
plt.yticks([])

plt.legend()
plt.show()

