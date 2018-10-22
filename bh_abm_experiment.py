from bh_abm import *
from generic_abm import *

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt

# Set the ABM Evaluation Budget
n_train_samples = 1000

# Set test size
n_test_samples = 10000

# Define Monte Carlo simulations
n_monte_carlo = 20

# Final test set size
n_final_test_samples = n_test_samples * n_monte_carlo

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

# Get parameters for train and test from the support
X_train, X_test = get_unirand_parameter_samples(n_train_samples, n_final_test_samples, bh_abm_exploration_range)

## REALVALUED

# Run the ABM on the train and test parameters in REALVALUED case...
y_train, y_test =  bh_abm_evaluate_samples(X_train, X_test, "real_valued")

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

## BINARY

# Run the ABM on the train and test parameters in BINARY case...
y_train, y_test =  bh_abm_evaluate_samples(X_train, X_test, "binary")

# Train a LogisticRegression model
LogisticRegression_model = fit_surrogate_as_logistic_classifier(X_train, y_train)

# Train a surrogate GaussianProcessClassifier model
GaussianProcessClassifier_model = fit_surrogate_as_gp_classifier(X_train, y_train)

# Train a surrogate GradientBoostingClassifier model
GradientBoostingClassifier_model = fit_surrogate_as_gbt_classifier(X_train, y_train)

# Evaluate the surrogates on the test set
y_hat_test = [None] * 3
y_hat_test[0] = LogisticRegression_model.predict(X_test)
y_hat_test[1] = GaussianProcessClassifier_model.predict(X_test)
y_hat_test[2] = GradientBoostingClassifier_model.predict(X_test)

# MSE performance
f1_performance = np.zeros((3, n_monte_carlo))
for sur_idx in range(len(y_hat_test)):
    for i in range(n_monte_carlo):
        f1_performance[sur_idx, i] = f1_score(y_test[i * n_test_samples:(i + 1) * n_test_samples], y_hat_test[int(sur_idx)][i * n_test_samples:(i + 1) * n_test_samples])
        
# Plot the Monte Carlo performance densities for each of the methods
experiment_labels = ["LogisticRegression", "GaussianProcessClassifier", "GradientBoostingClassifier"]

f1_performance = pd.DataFrame(f1_performance, index = experiment_labels)

LogisticRegression_label = "LogisticRegression: Mean " + '{:2.5f}'.format(f1_performance.iloc[0, :].mean()) + ", Variance " + '{:2.5f}'.format(f1_performance.iloc[0, :].var())
GaussianProcessClassifier_label = "GaussianProcessClassifier: Mean " + '{:2.5f}'.format(f1_performance.iloc[1, :].mean()) + ", Variance " + '{:2.5f}'.format(f1_performance.iloc[1, :].var())
GradientBoostingClassifier_label = "GradientBoostingClassifier: Mean " + '{:2.5f}'.format(f1_performance.iloc[2, :].mean()) + ", Variance " + '{:2.5f}'.format(f1_performance.iloc[2, :].var())

fig, ax = plt.subplots(figsize=(12, 5))
sns.distplot(f1_performance.iloc[0, :], label = LogisticRegression_label, ax = ax)
sns.distplot(f1_performance.iloc[1, :], label = GaussianProcessClassifier_label, ax = ax)
sns.distplot(f1_performance.iloc[2, :], label = GradientBoostingClassifier_label, ax = ax)

plt.title("Prediction Performance")
plt.xlabel('F1 Score')
plt.yticks([])

plt.legend()
plt.show()