from bh_abm import *
from generic_abm import *

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm
import seaborn as sns
import matplotlib.pyplot as plt

def bh_abm_budget_simulator(budget):

    # Set the ABM Evaluation Budget
    n_train_samples = budget

    # Set test size
    n_test_samples = 10000

    # Define Monte Carlo simulations
    n_monte_carlo = 100

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

    # Train a surrogate GradientBoostingRegressor model
    GradientBoostingRegressor_model = fit_surrogate_as_gbt_regression(X_train, y_train)
    
    # Evaluate the surrogate on the test set
    y_hat_test = GradientBoostingRegressor_model.predict(X_test)
    
    # MSE performance
    mse_performance = np.zeros(n_monte_carlo)
    for i in range(n_monte_carlo):
        mse_performance[i] = mean_squared_error(y_test[i * n_test_samples:(i + 1) * n_test_samples], y_hat_test[i * n_test_samples:(i + 1) * n_test_samples])
    mse_performance_mean = mse_performance.mean()
    mse_performance_sigma = np.std(mse_performance)
    mse_performance_95 = np.ptp(norm.interval(0.95, loc = mse_performance_mean, scale = mse_performance_sigma))

    ## BINARY

    # Run the ABM on the train and test parameters in BINARY case...
    y_train, y_test =  bh_abm_evaluate_samples(X_train, X_test, "binary")

    # Train a surrogate GradientBoostingClassifier model
    GradientBoostingClassifier_model = fit_surrogate_as_gbt_classifier(X_train, y_train)

    # Evaluate the surrogates on the test set
    y_hat_test = GradientBoostingClassifier_model.predict(X_test)

    # MSE performance
    f1_performance = np.zeros(n_monte_carlo)
    for i in range(n_monte_carlo):
        f1_performance[i] = f1_score(y_test[i * n_test_samples:(i + 1) * n_test_samples], y_hat_test[i * n_test_samples:(i + 1) * n_test_samples])
    f1_performance_mean = f1_performance.mean()
    f1_performance_sigma = np.std(f1_performance)
    f1_performance_95 = np.ptp(norm.interval(0.95, loc = f1_performance_mean, scale = f1_performance_sigma))
    
    return budget, mse_performance_mean, mse_performance_95, f1_performance_mean, f1_performance_95

def bh_abm_multiple_budgets(budgets):
    
    data = []
    for budget in budgets:
        new_data = bh_abm_budget_simulator(budget)
        
        data.append(new_data)
        
        print "Budget:", budget, "completed."
    
    return pd.DataFrame(data, columns = ["Budget", "MSE_mean", "MSE_95", "F_1_mean", "F_1_95"])
    
test = bh_abm_multiple_budgets([250, 500, 750, 1000, 1250, 1500, 1750, 2000])

fig, ax = plt.subplots(figsize=(12, 5))
plt.errorbar(x = np.array(test["Budget"]), y = np.array(test["MSE_mean"]), yerr = np.array(test["MSE_95"]))
plt.title("Mean Squared Errors")
plt.xlabel("Budget")
plt.ylabel("MSE")
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
plt.errorbar(x = np.array(test["Budget"]), y = np.array(test["F_1_mean"]), yerr = np.array(test["F_1_95"]))
plt.title("F_1 scores")
plt.xlabel("Budget")
plt.ylabel("F_1 Score")
plt.show()