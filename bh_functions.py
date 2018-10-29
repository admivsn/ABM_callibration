""" Ignore Warnings """
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

""" Imports """
import numpy as np
import pandas as pd
import sobol_seq
from scipy.stats.distributions import entropy
from scipy.stats import ks_2samp
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV # Cross validation


""" surrogate models """
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gaussian Process Regression (Kriging)
# modified version of kriging to make a fair comparison with regard
# to the number of hyperparameter evaluations
from sklearn.gaussian_process import GaussianProcessRegressor


""" cross-validation
Cross validation is used in each of the rounds to approximate the selected 
surrogate model over the data samples that are available. 
The evaluated parameter combinations are randomly split into two sets. An 
in-sample set and an out-of-sample set. The surrogate is trained and its 
parameters are tuned to an in-sample set, while the out-of-sample performance 
is measured (using a selected performance metric) on the out-of-sample set. 
This out-of-sample performance is then used as a proxy for the performance 
on the full space of unevaluated parameter combinations. In the case of the 
proposed procedure, this full space is approximated by the randomly selected 
pool.
"""

""" performance metric """
# Mean Squared Error
from sklearn.metrics import mean_squared_error, f1_score

""" Defaults Algorithm Tuning Constants """
_N_EVALS = 10
_N_SPLITS = 5
_CALIBRATION_THRESHOLD = 1.00
_P_VALUE_REJECT = 0.05

_TIME_STEPS = 100

""" Functions """
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def evaluate_bh_on_set(parameter_combinations):
    y = np.zeros(parameter_combinations.shape[0])
    num_params = parameter_combinations.shape[1]
    
    real_data = get_BH_real_data()
    
    if num_params == 9:
        for i, (trend_2, trend_1, switching_parameter, alpha, bias_2, weight_past_profits, bias_1, rational_expectation_cost, risk_free_return) in enumerate(parameter_combinations):
            response = BH_abm(trend_2 = trend_2,
                              trend_1 = trend_1,
                              switching_parameter = switching_parameter,
                              alpha = alpha,
                              bias_2 = bias_2,
                              weight_past_profits = weight_past_profits,
                              bias_1 = bias_1,
                              rational_expectation_cost = rational_expectation_cost,
                              risk_free_return = risk_free_return)
            y[i] = BH_constraint(response, real_data)

    return y

def get_BH_real_data():
    """C-Support Vector Classification.
    Parameters
    ----------
    C : float, optional (default=1.0)
        penalty parameter C of the error term.
    kernel : string, optional
         Description of this members.
    Attributes
    ----------
    `bar_` : array-like, shape = [n_features]
        Brief description of this attribute.
    Examples
    --------
    >>> clf = Foo()
    >>> clf.fit()
    []
    See also
    --------
    OtherClass
    """
    """ Get real data sample """
    data_close = pd.read_csv("sp500.csv")
    data_close.index = data_close['Date']
    data_close.drop('Date', axis=1, inplace=True)

    sample = data_close[-_TIME_STEPS:]
    sample = np.log(sample['Adj Close']).diff(1).dropna()

    return sample

def BH_abm(trend_2, 
           trend_1,
           switching_parameter, 
           alpha, 
           bias_2, 
           weight_past_profits,
           bias_1, 
           rational_expectation_cost, 
           risk_free_return, 
           T=_TIME_STEPS,
           _RNG_SEED=0):
    """C-Support Vector Classification.
    Parameters
    ----------
    C : float, optional (default=1.0)
        penalty parameter C of the error term.
    kernel : string, optional
         Description of this members.
    Attributes
    ----------
    `bar_` : array-like, shape = [n_features]
        Brief description of this attribute.
    Examples
    --------
    >>> clf = Foo()
    >>> clf.fit()
    []
    See also
    --------
    OtherClass
    """
    
    """ Default Response Value """
    response = np.array([0.0])

    """ Fixed Parameters (used inside positive price constraint) """
    share_type_1 = 0.5
    init_pdev_fund = 0.2

    """Set Fixed Parameters"""
    dividend_stream = 0.8
    dividend = dividend_stream
    init_wtype_1 = 0
    init_wtype_2 = 0
    sigma2 = 0.01
    fund_price = dividend_stream / (risk_free_return - 1)
    """ Check that the price is positive """
    if (share_type_1 * (trend_1 * init_pdev_fund + bias_1)) + ((1 - share_type_1) * (trend_2 * init_pdev_fund + bias_2)) > 0:
        """ Set RNG Seed"""
        np.random.seed(_RNG_SEED)
        random_dividend = np.random.uniform(low = -0.1, high = 0.1, size = T)

        """ Preallocate Containers """
        X = np.zeros(T)
        P = np.zeros(T)
        N1 = np.zeros(T)

        """ Run simulation """
        for time in range(T):
            # Update fraction of share_type_2
            share_type_2 = 1 - share_type_1

            # Produce Forecast
            forecast = share_type_1 * (trend_1 * init_pdev_fund + bias_1) + share_type_2 * (trend_2 * init_pdev_fund + bias_2)

            # Realized equilibrium_price
            equilibrium_price_realized = forecast / risk_free_return

            # Accumulated type 1 profits
            init_wtype_1 = weight_past_profits * init_wtype_1 + (
                                                                    equilibrium_price_realized - risk_free_return * init_pdev_fund) * (
                                                                    trend_1 * dividend + bias_1 - risk_free_return * init_pdev_fund) / (
                                                                    alpha * sigma2) - rational_expectation_cost

            # Accumulated type 2 profits
            init_wtype_2 = weight_past_profits * init_wtype_2 + (
                                                                    equilibrium_price_realized - risk_free_return * init_pdev_fund) * (
                                                                    trend_2 * dividend + bias_2 - risk_free_return * init_pdev_fund) / (
                                                                    alpha * sigma2)

            # Update fractions
            share_type_1 = np.exp(switching_parameter * init_wtype_1) / (np.exp(switching_parameter * init_wtype_1) + np.exp(switching_parameter * init_wtype_2))
            share_type_2 = 1 - share_type_1

            # Set initial conditions for next period
            dividend = init_pdev_fund
            random_dividend_fluctuation = random_dividend[time]
            init_pdev_fund = equilibrium_price_realized + random_dividend_fluctuation

            # set constraint on unstable diverging behaviour
            if init_pdev_fund > 100:
                init_pdev_fund = np.nan
            elif init_pdev_fund < 0:
                init_pdev_fund = np.nan

            """ Record Results """
            # Prices
            X[time] = init_pdev_fund
            P[time] = X[time] + fund_price
            N1[time] = share_type_1

        if X[~np.isnan(X)].shape[0] == T:
            response = np.diff(np.log(X[~np.isnan(X)]))
            
    return response

def BH_constraint(sim_series, sample):
    """
    """
    p_value = 0.0 # Reject LOW P VALUE = REJECT NULL HYPOTHESIS THAT DISTS ARE EQUAL

    # Check that the length is equal to the number of returns
    if sim_series.size == _TIME_STEPS - 1:
        
        np.random.seed(0)

        D, p_value = ks_2samp(sample, sim_series)

    return p_value

def set_surrogate_as_gbt_reg():
    """ 
    Set the surrogate model as Gradient Boosted Decision Trees. A helper 
    function to set the surrogate model and parameter space as Gradient Boosted 
    Decision Trees.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    surrogate_parameter_space :
        The parameter space of the surrogate model to be explored.
    """

    # Set the surrogate model as GradientBoostingRegressorm loss is a combination of L1 and L2 regularization
    surrogate_model = GradientBoostingRegressor(random_state = 0)
    
    # Define the parameter space of the surrogate model
    surrogate_parameter_space = {"n_estimators": randint(100, 900), # n_estimators
                                 "learning_rate": uniform(0.01, 0.99),   # learning_rate
                                 "max_depth": randint(10, 990),  # max_depth
                                 "subsample": uniform(0.25, 0.75)} # subsample

    return surrogate_model, surrogate_parameter_space

def set_surrogate_as_gbt_cla():
    """ 
    Set the surrogate model as Gradient Boosted Decision Trees. A helper 
    function to set the surrogate model and parameter space as Gradient Boosted 
    Decision Trees.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    surrogate_parameter_space :
        The parameter space of the surrogate model to be explored.
    """

    # Set the surrogate model as GradientBoostingRegressorm loss is a combination of L1 and L2 regularization
    surrogate_model = GradientBoostingClassifier(random_state = 0)
    
    # Define the parameter space of the surrogate model
    surrogate_parameter_space = {"n_estimators": randint(100, 900), # n_estimators
                                 "learning_rate": uniform(0.01, 0.99),   # learning_rate
                                 "max_depth": randint(10, 990),  # max_depth
                                 "subsample": uniform(0.25, 0.75)} # subsample

    return surrogate_model, surrogate_parameter_space

def custom_metric_regression(y_hat, y):
    return 'MSE', mean_squared_error(y.get_label(), y_hat)

def custom_metric_binary(y_hat, y):
    return 'MSE', f1_score(y.get_label(), y_hat, average='weighted')

def fit_surrogate_model(X, y):
    """ 
    Fit a surrogate model to the X,y parameter combinations in the real valued case.
    
    Parameters
    ----------
    X :
        Parameter combinations to train the model on.
    y :
        Output of the abm for these parameter combinations.
        
    Returns
    -------
    surrogate_model_fitted : 
        A surrogate model fitted.
    """
    
    # Get the surrogate model and parameter space
    surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt_reg()
    
    # Run randomized parameter search
    random_search = RandomizedSearchCV(surrogate_model, 
                                       param_distributions = surrogate_parameter_space,
                                       n_iter = 10, 
                                       cv = 5,
                                       random_state = 0)
    surrogate_model_tuned = random_search.fit(X, y)
        
    # Set the hyperparameters  of the surrograte model to the optimised hyperparameters from the random search
    surrogate_model.set_params(n_estimators = surrogate_model_tuned.best_params_["n_estimators"],
                               learning_rate = surrogate_model_tuned.best_params_["learning_rate"],
                               max_depth = surrogate_model_tuned.best_params_["max_depth"],
                               subsample = surrogate_model_tuned.best_params_["subsample"])

    # Fit the surrogate model
    surrogate_model.fit(X, y)

    return surrogate_model

def fit_entropy_classifier(X, y, calibration_threshold):
    """ 
    Fit a surrogate model to the X,y parameter combinations in the binary case.
    
    Parameters
    ----------
    X :
        Parameter combinations to train the model on.
    y :
        Output of the abm for these parameter combinations (0 or 1).
        
    Returns
    -------
    surrogate_model_fitted : 
        A surrogate model fitted.
    """
    y_binary = calibration_condition(y, calibration_threshold)
    clf, surrogate_parameter_space = set_surrogate_as_gbt_cla()
    
    # Run randomized parameter search
    random_search = RandomizedSearchCV(surrogate_model, 
                                       param_distributions = surrogate_parameter_space,
                                       n_iter = 10, 
                                       cv = 5,
                                       random_state = 0)
    clf_tuned = random_search.fit(X, y_binary)
    
    # Set the hyperparameters  of the surrograte model to the optimised hyperparameters from the random search
    clf.set_params(n_estimators = surrogate_model_tuned.best_params_["n_estimators"],
                               learning_rate = surrogate_model_tuned.best_params_["learning_rate"],
                               max_depth = surrogate_model_tuned.best_params_["max_depth"],
                               subsample = surrogate_model_tuned.best_params_["subsample"])
    
    # Fit the surrogate model
    clf.fit(X, y_binary)
    
    return clf

def get_sobol_samples(n_dimensions, samples, parameter_support):
    """
    """
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = sobol_seq.i4_sobol_generate(n_dimensions, samples)

    # Compute the parameter mappings between the Sobol samples and supports
    sobol_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return sobol_samples

def get_unirand_samples(n_dimensions, samples, parameter_support):
    """
    """
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = np.random.rand(n_dimensions,samples).T

    # Compute the parameter mappings between the Sobol samples and supports
    unirand_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return unirand_samples

def get_round_selections(evaluated_set_X, evaluated_set_y,
                         unevaluated_set_X,
                         predicted_positives, num_predicted_positives,
                         samples_to_select, calibration_threshold,
                         budget):
    """
    """
    samples_to_select = np.min([abs(budget - evaluated_set_y.shape[0]),
                                samples_to_select]).astype(int)

    if num_predicted_positives >= samples_to_select:
        round_selections = int(samples_to_select)
        selections = np.where(predicted_positives == True)[0]
        selections = np.random.permutation(selections)[:round_selections]

    elif num_predicted_positives <= samples_to_select:
        # select all predicted positives
        selections = np.where(predicted_positives == True)[0]

        # select remainder according to entropy weighting
        budget_shortfall = int(samples_to_select - num_predicted_positives)

        selections = np.append(selections,
                               get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                                                      unevaluated_set_X,
                                                      calibration_threshold,
                                                      budget_shortfall))

    else:  # if we don't have any predicted positive calibrations
        selections = get_new_labels_entropy(clf, unevaluated_set_X, samples_to_select)

    to_be_evaluated = unevaluated_set_X[selections]
    unevaluated_set_X = np.delete(unevaluated_set_X, selections, 0)
    evaluated_set_X = np.vstack([evaluated_set_X, to_be_evaluated])
    evaluated_set_y = np.append(evaluated_set_y, evaluate_bh_on_set(to_be_evaluated))

    return evaluated_set_X, evaluated_set_y, unevaluated_set_X

def get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                           unevaluated_X, calibration_threshold,
                           number_of_new_labels):
    """ Get a set of parameter combinations according to their predicted label entropy
    """
    clf = fit_entropy_classifier(evaluated_set_X, evaluated_set_y, calibration_threshold)

    y_hat_probability = clf.predict_proba(unevaluated_X)
    y_hat_entropy = np.array(map(entropy, y_hat_probability))
    y_hat_entropy /= y_hat_entropy.sum()
    unevaluated_X_size = unevaluated_X.shape[0]

    selections = np.random.choice(a=unevaluated_X_size,
                                  size=number_of_new_labels,
                                  replace=False,
                                  p=y_hat_entropy)
    return selections

print ("Imported successfully")