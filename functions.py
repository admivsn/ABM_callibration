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

""" Functions """
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def evaluate_islands_on_set(parameter_combinations):
    y = np.zeros(parameter_combinations.shape[0])
    num_params = parameter_combinations.shape[1]

    if num_params == 1:
        for i, rho in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)
    elif num_params == 2:
        for i, (rho, alpha) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 3:
        for i, (rho, alpha, phi) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 4:
        for i, (rho, alpha, phi, pi) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             pi=pi, _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 5:
        for i, (rho, alpha, phi, pi, eps) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             pi=pi, eps=eps, _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    return y

def island_abm(rho=0.01,
               alpha=1.5,
               phi=0.4,
               pi=0.4,
               eps=0.1,
               lambda_param=1,
               T=100,
               N=50,
               _RNG_SEED=0):
    """ Islands growth model
    Parameters
    ----------
    rho :
    alpha :
    phi : float, required
    eps :
    lambda_param: (Default = 1)
    T : int, required
    The number of periods for the simulation
    N : int, optional (Default = 50)
    Number of firms
    _RNG_SEED : int, optional (Default = 0)
    Random number seen
    Output
    ------
    GDP : array, length = [,T]
    Simulated GPD
    """
    # Set random number seed
    np.random.seed(_RNG_SEED)

    T_2 = int(T / 2)

    GDP = np.zeros((T, 1))

    # Distributions
    # Precompute random binomial draws
    xy = np.random.binomial(1, pi, (T, T))
    xy[T_2, T_2] = 1

    # Containers
    s = np.zeros((T, T))
    A = np.ones((N, 6))

    # Initializations
    A[:, 1] = T_2
    A[:, 2] = T_2
    m = np.zeros((T, T))
    m[T_2, T_2] = N
    dest = np.zeros((N, 2))

    """ Begin ABM Code """
    for t in range(T):
        w = np.zeros((N, N))
        signal = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    if A[j, 0] == 1:
                        w[i, j] = np.exp(-rho * (np.abs(A[j, 1] - A[i, 1]) + \
                                                 np.abs(A[j, 2] - A[i, 2])))

                        if np.random.rand() < w[i, j]:
                            signal[i, j] = s[int(A[j, 1]), int(A[j, 2])]

            if A[i, 0] == 1:
                A[i, 4] = s[int(A[i, 1]), int(A[i, 2])] * \
                          m[int(A[i, 1]), int(A[i, 2])] ** alpha
                A[i, 3] = s[int(A[i, 1]), int(A[i, 2])]

            if A[i, 0] == 3:
                A[i, 4] = 0
                rnd = np.random.rand()
                if rnd <= 0.25:
                    A[i, 1] += 1
                else:
                    if rnd <= 0.5:
                        A[i, 1] -= 1
                    else:
                        if rnd <= 0.75:
                            A[i, 2] += 1
                        else:
                            A[i, 2] -= 1

                if xy[int(A[i, 1]), int(A[i, 2])] == 1:
                    A[i, 0] = 1
                    m[int(A[i, 1]), int(A[i, 2])] += 1
                    if m[int(A[i, 1]), int(A[i, 2])] == 1:
                        s[int(A[i, 1]), int(A[i, 2])] = \
                            (1 + int(np.random.poisson(lambda_param))) * \
                            (A[i, 1] + A[i, 2]) + phi * A[i, 5] + np.random.randn()

            if (A[i, 0] == 1) and (np.random.rand() <= eps):
                A[i, 0] = 3
                A[i, 5] = A[i, 4]
                m[int(A[i, 1]), int(A[i, 2])] -= 1

            if t > T / 100:
                if A[i, 0] == 2:
                    A[i, 4] = 0
                    if dest[i, 0] != A[i, 1]:
                        if dest[i, 0] > A[i, 1]:
                            A[i, 1] += 1
                        else:
                            A[i, 1] -= 1
                    else:
                        if dest[i, 1] != A[i, 2]:
                            if dest[i, 1] > A[i, 2]:
                                A[i, 2] += 1
                            else:
                                A[i, 2] -= 1
                    if (dest[i, 0] == A[i, 1]) and (dest[i, 1] == A[i, 2]):
                        A[i, 0] = 1
                        m[int(dest[i, 0]), int(dest[i, 1])] += 1
                if A[i, 0] == 1:
                    best_sig = np.max(signal[i, :])
                    if best_sig > s[int(A[i, 1]), int(A[i, 2])]:
                        A[i, 0] = 2
                        A[i, 5] = A[i, 4]
                        m[int(A[i, 1]), int(A[i, 2])] -= 1
                        index = np.where(signal[i, :] == best_sig)[0]
                        if index.shape[0] > 1:
                            ind = int(index[int(np.random.uniform(0, len(index)))])
                        else:
                            ind = int(index)
                        dest[i, 0] = A[ind, 1]
                        dest[i, 1] = A[ind, 2]

        GDP[t, 0] = np.sum(A[:, 4])

    log_GDP = np.log(GDP)

    return log_GDP

def calibration_measure(log_GDP):
    """ Calibration Measure
    Input
    -----
    GDP : array, required, length = [,T]
    Output
    ------
    agdp : float
    Average GDP growth rate
    """
    T = log_GDP.shape[0]
    log_GDP = log_GDP[~np.isinf(log_GDP)]
    log_GDP = log_GDP[~np.isnan(log_GDP)]
    if log_GDP.shape[0] > 0:
        GDP_growth_rate = (log_GDP[-1] - log_GDP[0]) / T
    else:
        GDP_growth_rate = 0

    return GDP_growth_rate

def calibration_condition(average_GDP_growth_rate, threshold_condition):
    return average_GDP_growth_rate >= threshold_condition

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
    evaluated_set_y = np.append(evaluated_set_y, evaluate_islands_on_set(to_be_evaluated))

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