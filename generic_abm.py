###############################################################################
############################# IMPORTS / WARNINGS ##############################
###############################################################################

import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV # Cross validation
from sklearn.metrics import mean_squared_error, f1_score # Performance metric for real and binary outcomes

###############################################################################
############################## GENERIC FUNCTIONS ##############################
###############################################################################

def get_unirand_parameter_samples(n_train_samples, n_test_samples, parameter_support):
    """
    Generate n_train_samples and n_test_samples from the parameter support.
    
    Parameters
    ----------    
    n_train_samples:
        The number of training samples that you wish to generate.
    n_test_samples:
        The number of test samples that you wish to generate.  
    parameter_support:
        The support of each parameter that you wish to explore.
        
    Returns
    -------
    unirand_train_samples:
        Training samples.
    unirand_test_samples:
        Test samples.
    
    """
    # Get the range for the support, array containing the range of each parameter
    support_range = parameter_support[:, 1] - parameter_support[:, 0]
    
    # Generate all of the samples
    random_samples = np.random.rand(parameter_support.shape[0], n_train_samples + n_test_samples).T
    unirand_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])
    
    # Split them into train and test accordingly
    unirand_train_samples = unirand_samples[0: n_train_samples, ]
    unirand_test_samples = unirand_samples[n_train_samples:, ]

    return unirand_train_samples, unirand_test_samples

def custom_metric_regression(y_hat, y):
    """
    The performance metric in the real valued case.
    
    Parameters
    ----------
    y_hat:
        The fitted y values from the surrogate model.
    y:
        The real y values simulated directly from the abm.
        
    Returns
    -------
    MSE:
        The mean_squared_error of the y_hat and y values.
    """
    return 'MSE', mean_squared_error(y.get_label(), y_hat)

def custom_metric_binary(y_hat, y):
    """
    The performance metric in the binary case.
    
    Parameters
    ----------
    y_hat:
        The fitted y values from the surrogate model.
    y:
        The real y values simulated directly from the abm.
        
    Returns
    -------
    MSE:
        The f1_score of the y_hat and y values.
    """
    return 'F1', f1_score(y.get_label(), y_hat, average = 'weighted')

###############################################################################
########################## Linear/Logistic Regression #########################
###############################################################################
    
def set_surrogate_as_linear():
    """ 
    Set the surrogate model as Linear Regression.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    """

    # Set the surrogate model as Linear Regression
    surrogate_model = LinearRegression()

    return surrogate_model

def set_surrogate_as_logistic():
    """ 
    Set the surrogate model as Logistic Regression.

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

    # Set the surrogate model as Logistic Regression
    surrogate_model = LogisticRegression(random_state = 0)
    
    # Define the parameter space of the surrogate model
    surrogate_parameter_space = {"penalty": ["l1", "l2"],
                                 "max_iter": uniform(10, 190)}

    return surrogate_model, surrogate_parameter_space

def fit_surrogate_as_linear_regression(X, y):
    """
    Set the surrogate model as LogisticRegression.
    
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
    
    # Get the surrogate model
    surrogate_model = set_surrogate_as_linear()

    # Fit the surrogate model
    surrogate_model.fit(X, y)

    return surrogate_model    
    

def fit_surrogate_as_logistic_classifier(X, y):
    """
    Set the surrogate model as LogisticRegression.
    
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
    surrogate_model, surrogate_parameter_space = set_surrogate_as_logistic()
    
    # Run randomized parameter search
    random_search = RandomizedSearchCV(surrogate_model, 
                                       param_distributions = surrogate_parameter_space,
                                       n_iter = 10, 
                                       cv = 5,
                                       random_state = 0)
    surrogate_model_tuned = random_search.fit(X, y)
        
    # Set the hyperparameters  of the surrograte model to the optimised hyperparameters from the random search
    surrogate_model.set_params(penalty = surrogate_model_tuned.best_params_["penalty"],
                               max_iter = surrogate_model_tuned.best_params_["max_iter"])

    # Fit the surrogate model
    surrogate_model.fit(X, y)

    return surrogate_model

###############################################################################
######################### GAUSSIAN PROCESS REGRESSION #########################
###############################################################################

def fit_surrogate_as_gp_regression(X, y):
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
    
    # Get the surrogate model
    surrogate_model = GaussianProcessRegressor(random_state = 0)

    # Fit the surrogate model
    surrogate_model.fit(X, y)

    return surrogate_model

def fit_surrogate_as_gp_classifier(X, y):
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
    
    # Get the surrogate model
    surrogate_model = GaussianProcessClassifier(random_state = 0)

    # Fit the surrogate model
    surrogate_model.fit(X, y)

    return surrogate_model

###############################################################################
####################### Gradient Boosted Decision Trees #######################
###############################################################################

def set_surrogate_as_gbt_regressor():
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

def set_surrogate_as_gbt_classifier():
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

def fit_surrogate_as_gbt_regression(X, y):
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
    surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt_regressor()
    
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

def fit_surrogate_as_gbt_classifier(X, y):
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
    
    # Get the surrogate model and parameter space
    surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt_classifier()
    
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
