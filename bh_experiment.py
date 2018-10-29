from bh_functions import *

# Set the ABM Evaluation Budget
budget = 200

# Set out-of-sample test and montecarlo sizes
test_size = 1000
montecarlos = 10

# Get an on out-of-sample test set that does not have combinations from the
# batch or iterative experiments
final_test_size = (test_size * montecarlos)

# Set the ABM parameters and support
bh_exploration_range = np.array([(-2.0, 2.0), # trend_2
                                 (-2.0, 2.0), # trend_1
                                 (0.0, 10.0), # switching_parameter
                                 (0.0, 100.0),# alpha 
                                 (-2.0, 2.0), # bias_2 
                                 (0.0, 1.0),  # weight_past_profits
                                 (-2.0, 2.0), # bias_1
                                 (0.0, 5.0),  # rational_expectation_cost
                                 (1.01, 1.1)])# risk_free_return

param_dims = bh_exploration_range.shape[0]

load_data = False

if load_data: # ENSURE THAT THE FILES ARE SAVED ALREADY
    evaluated_set_X_batch = pd.read_csv("_bh_budget_" + str(budget) + "_X.csv", index_col = 0).values
    evaluated_set_y_batch = pd.read_csv("_bh_budget_" + str(budget) + "_y.csv", index_col = 0).values
    oos_set = pd.read_csv("_bh_budget_" + str(budget) + "_X_oos.csv", index_col = 0).values
    y_test = pd.read_csv("_bh_budget_" + str(budget) + "_y_oos.csv",index_col=0).values
else:
    # Generate Sobol samples for training set
    n_dimensions = bh_exploration_range.shape[0]
    
    evaluated_set_X_batch = get_sobol_samples(n_dimensions, budget, bh_exploration_range)
    evaluated_set_y_batch = evaluate_bh_on_set(evaluated_set_X_batch)
    
    pd.DataFrame(evaluated_set_X_batch).to_csv("_bh_budget_" + str(budget) + "_X.csv")
    pd.DataFrame(evaluated_set_y_batch).to_csv("_bh_budget_" + str(budget) + "_y.csv")
    
    # Build Out-of-sample set
    oos_set = get_unirand_samples(n_dimensions, final_test_size*budget, bh_exploration_range)
    selections = []
    for i, v in enumerate(oos_set):
        if (v not in evaluated_set_X_batch):
            selections.append(i)
    oos_set = unique_rows(oos_set[selections])[:final_test_size]

    # Evaluate the test set for the ABM response
    y_test = evaluate_bh_on_set(oos_set)   
    
    pd.DataFrame(oos_set).to_csv("_bh_budget_" + str(budget) + "_X_oos.csv")
    pd.DataFrame(y_test).to_csv("_bh_budget_" + str(budget) + "_y_oos.csv")
    
# Fit the GaussianProcessRegressor model
surrogate_models_GaussianProcessRegressor = GaussianProcessRegressor(random_state = 0)
surrogate_models_GaussianProcessRegressor.fit(evaluated_set_X_batch, evaluated_set_y_batch)

# Fit the GradientBoostingRegressor
surrogate_model_GradientBoostingRegressor = fit_surrogate_model(evaluated_set_X_batch, evaluated_set_y_batch)

# Make predictions from each model
y_hat_test = [None] * 2
y_hat_test[0] = surrogate_models_GaussianProcessRegressor.predict(oos_set)
y_hat_test[1] = surrogate_model_GradientBoostingRegressor.predict(oos_set)

# Output y_hat (predictions of each model) as a csv
pd.DataFrame(y_hat_test[0]).to_csv("_bh_budget_" + str(budget) + "_y_hat_GaussianProcessRegressor.csv")
pd.DataFrame(y_hat_test[1]).to_csv("_bh_budget_" + str(budget) + "_y_hat_GradientBoostingRegressor.csv")

# MSE performance
mse_perf = np.zeros((2, montecarlos))
for sur_idx in range(len(y_hat_test)):
    for i in range(montecarlos):
        mse_perf[sur_idx, i] = mean_squared_error(y_test[i * test_size:(i + 1) * test_size],
                                                  y_hat_test[int(sur_idx)][i * test_size:(i + 1) * test_size])
        

experiment_labels = ["GaussianProcessRegressor", "GradientBoostingRegressor"]

mse_perf = pd.DataFrame(mse_perf, index = experiment_labels)

GaussianProcessRegressor_label = "GaussianProcessRegressor: Mean " + '{:2.5f}'.format(mse_perf.iloc[0, :].mean()) + ", Variance " + '{:2.5f}'.format(mse_perf.iloc[0, :].var())
GradientBoostingRegressor_label = "GradientBoostingRegressor: Mean " + '{:2.5f}'.format(mse_perf.iloc[1, :].mean()) + ", Variance " + '{:2.5f}'.format(mse_perf.iloc[1, :].var())

fig, ax = plt.subplots(figsize=(12, 5))
sns.distplot(mse_perf.iloc[0, :], label = GaussianProcessRegressor_label, ax = ax)
sns.distplot(mse_perf.iloc[1, :], label = GradientBoostingRegressor_label, ax = ax)

plt.title("Out-Of-Sample Prediction Performance")
plt.xlabel('Mean-Squared Error')
plt.yticks([])

plt.legend()

fig.savefig("_budget_" + str(budget) + "_bh_GradientBoostingRegressor_GaussianProcessRegressor.png");