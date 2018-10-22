###############################################################################
############################# IMPORTS / WARNINGS ##############################
###############################################################################

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt

###############################################################################
################################ MAIN ABM CODE ################################
###############################################################################

def bh_abm(beta = 1.0,
           n_1 = 0.5,
           b_1 = 0.5,
           b_2 = 0.2,
           g_1 = 1.0, 
           g_2 = 1.0,
           C = 0.1,
           w = 0.1,
           sigma = 0.1,
           v = 1,
           r = 0.1,
           T = 503,
           _RNG_SEED = 0):
    """
    Simulation of B&H asset pricing model.
    
    Parameters
    ----------
    beta :
        Intensity of choice [0, inf).
    n_1 : (Default = 0.5)
        Initial share of type 1 traders [0, 1].
    b_1 :
        Bias of type 1 traders (-inf, inf).
    b_2 :
        Bias of type 2 traders (-inf, inf).
    g_1:
        Trend component of type 1 traders (-inf, inf).
    g_2:
        Trend component of type 2 traders (-inf, inf).
    C :
        Cost of obtaining type 1 forecasts [0, inf).
    w :
        Weight to past profits [0, 1] 
    sigma :
        Asset volatility (0, inf).
    v :
        Attitude towards risk [0, inf].
    r :
        Risk-free return (1, inf).
    T : int, required
        Number of periods.
    _RNG_SEED : int, optional (Default = 0)
        Random number seen.
        
    Returns
    -------
    simulated_data: 
        Logarithmic return of the asset.
    """
    
    simulated_data = np.array([0.0])

    x_prev = 0.2 # Previous deviation from fundamental price
    
    y_bar = 0.1 # Expected dividend of risky asset
        
    R = 1 + r 
    
    p_star = y_bar / (R-1) # Fundamental price of asset that satisfys no bubbles
    
    # Check that R * x_t is going to be positive [x_t = p_t - p_t^* (deviation from fundamental price)]
    if (n_1 * (g_1 * x_prev + b_1)) + ((1 - n_1) * (g_2 * x_prev + b_2)) > 0:
                    
        # Set random seed
        np.random.seed(_RNG_SEED)
        random_dividend = np.random.uniform(low = -0.1, high = 0.1, size = T)

        # Preallocate Containers to store results
        X = np.zeros(T) # Deviations from fundamental prices
        P = np.zeros(T) # Prices (ex. divdend) of risky asset
        N_1 = np.zeros(T) # Array to store the share of type 1 traders
        
        # Performance measures initialised
        U_1, U_2 = 0, 0
        
        """ Run simulation """
        for t in range(T):
            
            # Define share of type 2 traders
            n_2 = 1 - n_1
            
            # Price x using the equilibrium condition
            x_equil = (n_1 * (g_1 * x_prev + b_1)) + (n_2 * (g_2 * x_prev + b_2)) / R
                        
            # Update the accumulated profits of each strategy *** DOUBLE CHECK HERE***
            pi_1 = ((x_equil - (R * x_prev)) * ((g_1 * x_prev) + b_1 - (R * x_prev))) / (v * sigma**2) - C
            pi_2 = ((x_equil - (R * x_prev)) * ((g_2 * x_prev) + b_2 - (R * x_prev))) / (v * sigma**2)
                        
            # Update the fitness measure of each strategy
            U_1 = pi_1 + (w * U_1)
            U_2 = pi_2 + (w * U_2)
            
            # Update the fractions of each strategy
            n_1 = np.exp(beta * U_1) / (np.exp(beta * U_1) + np.exp(beta * U_2))
            n_2 = 1 - n_1

            # Set initial conditions for next period
            x_prev = x_equil
            x = x_equil + random_dividend[t]

            # Set constraints on unstable diverging behaviour
            if x > 100:
                x = np.nan
            elif x < 0:
                x = np.nan

            """ Record Results """
            # Prices
            X[t] = x
            P[t] = x + p_star
            N_1[t] = n_1
            
        # If there are no nan values (diverging behaviour)...
        if X[~np.isnan(X)].shape[0] == T:
                        
            # Return the first order differences
            simulated_data = np.diff(np.log(X[~np.isnan(X)]))
            

    return simulated_data

###############################################################################
############################### CALIBRATION CODE ##############################
###############################################################################

def bh_abm_get_real_data():
    """ 
    Get real data sample from file named sp500.csv .
    
    Returns
    -------
    sample:
        Differenced real data sample.
    """
    
    # Read in data using pandas 
    data_close = pd.read_csv('/Users/b4017054/Documents/Work/Newcastle/PhD/ABM/sani/sp500.csv')
    # Set 'Date' as the index
    data_close.index = data_close['Date']
    
    # Sample the 'Adj Close' column, difference it by one, and drop any NA's
    sample = np.log(data_close['Adj Close']).diff(1).dropna()
    
    return sample

def bh_abm_real_valued_calibration_measure(simulated_data,
                                           real_data):
    """ 
    Real valued calibration measure for the B&H abm.
    
    Parameters
    ----------
    simulated_data: array
        GDP_growth_rate's of the simulated data (output of bh_abm_evaluate_on_set).
    real_data: array
        GDP growth rates of the real data (output of get_bh_abm_real_data).
    
    Returns
    -------
    p_value: float
        The p-value of the KS test as to wether the two distributions are equal.
    """
    
    # Set the default response
    p_value = 1.00 # (Reject)
    
    # Check that the length of the simulated data is equal to that of the real data returns
    if len(simulated_data) == len(real_data):
                
        # Set random seed
        np.random.seed(0)

        # Carry out the KS test
        response, p_value = ks_2samp(simulated_data, real_data)

    return p_value

def bh_abm_binary_calibration_measure(simulated_data,
                                      real_data):
    """ 
    Binary calibration measure for the B&H abm.
    
    Parameters
    ----------
    simulated_data: array
        GDP_growth_rate's of the simulated data (output of bh_abm_evaluate_on_set).
    real_data: array
        GDP growth rates of the real data (output of get_bh_abm_real_data).
    
    Returns
    -------
    response: 
        The binary response as to wether the two distributions are equal under the KS test (5% threshold).
        
    """
    
    # Set the default response
    response = 0.0 # (Reject)
        
    # Check that the length of the simulated data is equal to that of the real data returns
    if len(simulated_data) == len(real_data):
        
        # Set random seed
        np.random.seed(0)

        # Carry out the KS test
        response, p_value = ks_2samp(simulated_data, real_data)

        # Reject if p-value is less than 5%
        if p_value < 0.05:
            response = 1.0 # Accept it
        else:
            response = 0.0 # Reject it

    return response

###############################################################################
############################ RUN ON PARAMETER SETS ############################
###############################################################################
      
def bh_abm_on_set(parameter_combinations, calibration_measure):
    """
    Run bh_abm on a set of parameter combinations.
    
    Parameters
    ----------
    parameter_combinations: array
        Array of parameter combinations to run the bh_abm on.
    calibration_measure: ["real_valued", "binary"]
        The calibration measure that you wish to use.    
        
    Returns
    -------
    p_values: array 
        The p-values from the callibration measure of each parameter combination.
    responses: array
        The binary response associated to the p-value.
    """
    
    # Pre allocate array to store results in
    response = np.zeros(parameter_combinations.shape[0])
    
    # Get the real data to callibrate against
    real_data = bh_abm_get_real_data()

    for i, (beta, n_1, b_1, b_2, g_1, g_2, C, w, sigma, v, r) in enumerate(parameter_combinations):
                
        # Simulate the data for those parameters
        simulated_data = bh_abm(beta = beta, 
                                n_1 = n_1,
                                b_1 = b_1,
                                b_2 = b_2,
                                g_1 = g_1,
                                g_2 = g_2,
                                C = C,
                                w = w,
                                sigma = sigma,
                                v = v,
                                r = r)
                
        # Input the calibration metric into the array
        if calibration_measure == "real_valued":
            response[i] = bh_abm_real_valued_calibration_measure(simulated_data, real_data)
        if calibration_measure == "binary":
            response[i] = bh_abm_binary_calibration_measure(simulated_data, real_data)

    return response

def bh_abm_evaluate_samples(unirand_train_samples, 
                            unirand_test_samples,
                            calibration_measure):
    """ 
    Evaluate the B&H abm on the train and test samples provided.
    
    Parameters
    ----------
    unirand_train_samples: array
        Array of training parameter combinations to run the bh_abm on.
    unirand_test_samples: array
        Array of training parameter combinations to run the bh_abm on.
    calibration_measure: ["real_valued", "binary"]
        The calibration measure that you wish to use.
    
    Returns
    ------
    evaluated_Y_train: array
        Training labels for binary and real valued case.
    evaluated_Y_test: array
        Test labels for binary and real valued case.
    """
    evaluated_Y_train = bh_abm_on_set(unirand_train_samples, calibration_measure)
    
    evaluated_Y_test = bh_abm_on_set(unirand_test_samples, calibration_measure)
    
    return evaluated_Y_train, evaluated_Y_test

def bh_abm_make_plot(data, 
                         title):
    """ 
    Plot the distributions of the B&H ABM parameters
    
    Parameters
    ----------
    data: DataFrame
        Contains the parameter sets to be plotted.
    title: String
        Title of the plot.
        
    Returns
    ------
    plot:
        Plot of the parameter distributions.
        
    """
    
    # Plot the prior distributions of each parameter
    f, axes = plt.subplots(3, 4, figsize=(7, 7))
    f.suptitle(title)
    sns.distplot(data["beta"],                 ax=axes[0, 0], rug = True)
    sns.distplot(data["n_1"],                  ax=axes[0, 1], rug = True)
    sns.distplot(data["b_1"],                  ax=axes[0, 2], rug = True)
    sns.distplot(data["b_2"],                  ax=axes[0, 3], rug = True)
    sns.distplot(data["g_1"],                  ax=axes[1, 0], rug = True)
    sns.distplot(data["g_2"],                  ax=axes[1, 1], rug = True)
    sns.distplot(data["C"],                    ax=axes[1, 2], rug = True)
    sns.distplot(data["w"],                    ax=axes[1, 3], rug = True)
    sns.distplot(data["sigma"],                ax=axes[2, 0], rug = True)    
    sns.distplot(data["v"],                    ax=axes[2, 1], rug = True)
    sns.distplot(data["r"],                    ax=axes[2, 2], rug = True)
    f.tight_layout()
    plt.subplots_adjust(top = 0.9)
    
    return plt.show()

