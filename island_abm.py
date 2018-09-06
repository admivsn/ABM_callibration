import numpy as np
from scipy.stats import exponpow
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

def island_abm(rho = 0.01,
               lambda_param = 1,
               alpha = 1.5,
               phi = 0.4,
               pi = 0.4,
               eps = 0.1,
               N = 50,
               T = 100,
               _RNG_SEED = 0):    
    """ 
    Simulation of islands growth model.
    
    Parameters
    ----------
    rho : float
        Degree of locality in the diffusion of knowledge.
    alpha : float
        Productivity of labour in extraction.
    phi : float, required
        Cumulative learning effect.
    eps : float
        Willingness to explore.
    lambda_param: (Default = 1)
        Mean of Poisson RV (jumps in technology)
    T : int, required
        The number of periods for the simulation.
    N : int, optional (Default = 50)
        Number of firms
    _RNG_SEED : int, optional (Default = 0)
        Random number seen
        
    Returns
    -------
    GDP : array, length = [,T]
        Simulated GPD
    """
    # Set random number seed
    np.random.seed(_RNG_SEED)

    T_2 = int(T / 2)

    GDP = np.zeros(T)

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
                        
        GDP[t] = np.sum(A[:, 4])
    
    
    
    if GDP[2:].all() != 0.0:
        log_GDP = np.log(GDP)
    else:
        log_GDP = np.zeros(T)
    
    return log_GDP

def island_abm_real_valued_calibration_measure(log_GDP):
    """ 
    Real valued calibration measure for the Island abm.
    
    Parameters
    ----------
    log_GDP : array, required, length = [,T]
        Output of island_abm.
    
    Returns
    -------
    b:
        The value of b from the fitted exponential power distribution.
    """
    
    # Initialise the callibration output to fail
    b = 0
    
    # Get rid of inf and nan values
    log_GDP = log_GDP[~np.isinf(log_GDP)] 
    log_GDP = log_GDP[~np.isnan(log_GDP)]
    
    if log_GDP.shape[0] > 0:        

        # Fit a symmetric exponential power distribution
        b, mean, a = exponpow.fit(log_GDP) #?????

    return b

def island_abm_binary_calibration_measure(log_GDP):
    """ 
    Binary calibration measure for the Island abm.
    
    Parameters
    ----------
    log_GDP : array, required, length = [,T]
        Output of island_abm.
    
    Returns
    -------
    calibration: binary
        Wether or not the log_GDP passes the binary condition
        
    """
    
    # Initialise the callibration output to fail
    calibration = 0
    
    T = log_GDP.shape[0]
    # Get rid of inf and nan values
    log_GDP = log_GDP[~np.isinf(log_GDP)] 
    log_GDP = log_GDP[~np.isnan(log_GDP)]
    
    # Calculate average growth rate
    if log_GDP.shape[0] > 0:
        average_GDP_growth_rate = (log_GDP[-1] - log_GDP[0]) / T
        
        # Condition
        if average_GDP_growth_rate > 0.02:
            
            # Fit a symmetric exponential power distribution
            b, mean, a = exponpow.fit(log_GDP)
            
            # Condition
            if b > 1:
                calibration = 1

    return calibration

def island_abm_on_set(parameter_combinations, 
                      calibration_measure):
    """
    Run island_abm on a set of parameter combinations. for a given calibration measure
    
    Parameters
    ----------
    parameter_combinations: array
        Array of parameter combinations to run the bh_abm on.
    calibration_measure: ["real_valued", "binary"]
        The calibration measure that you wish to use.
    
    Returns
    -------
    response: array
        The responsed associated to the calibration measure.
    """
    
    # Pre allocate array to store results in
    response = np.zeros(parameter_combinations.shape[0])
    
    for i, (rho, lambda_param, alpha, phi, pi, eps) in enumerate(parameter_combinations):
                
        # Simulate the data for those parameters
        simulated_data = island_abm(rho = rho,
                                    lambda_param = lambda_param,
                                    alpha = alpha,
                                    phi = phi,
                                    pi = pi,
                                    eps = eps)
        
        # Input the calibration metric into the array
        if calibration_measure == "real_valued":
            response[i] = island_abm_real_valued_calibration_measure(simulated_data)
        if calibration_measure == "binary":
            response[i] = island_abm_binary_calibration_measure(simulated_data)

    return response

def island_abm_evaluate_samples(unirand_train_samples, 
                                unirand_test_samples,
                                calibration_measure):
    """ 
    Evaluate the Island abm on the train and test samples provided.
    
    Parameters
    ----------
    unirand_train_samples: array
        Array of training parameter combinations to run the island_abm on.
    unirand_test_samples: array
        Array of training parameter combinations to run the island_abm on.
    calibration_measure: ["real_valued", "binary"]
        The calibration measure that you wish to use.
    
    Returns
    ------
    evaluated_Y_train: array
        Training labels for the calibration_measure specified.
    evaluated_Y_test: array
        Test labels for the calibration_measure specified.

    """
    evaluated_Y_train = island_abm_on_set(unirand_train_samples, calibration_measure)
    
    evaluated_Y_test = island_abm_on_set(unirand_test_samples, calibration_measure)
    
    return evaluated_Y_train, evaluated_Y_test

def island_abm_make_plot(data, 
                         title):
    """ 
    Plot the distributions of the Island ABM parameters
    
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
    f, axes = plt.subplots(2, 3, figsize=(7, 7))
    f.suptitle(title)
    sns.distplot(data["rho"],             ax=axes[0, 0], rug = True)
    sns.distplot(data["lambda_param"],    ax=axes[0, 1], rug = True)
    sns.distplot(data["alpha"],           ax=axes[0, 2], rug = True)
    sns.distplot(data["phi"],             ax=axes[1, 0], rug = True)
    sns.distplot(data["pi"],              ax=axes[1, 1], rug = True)
    sns.distplot(data["eps"],             ax=axes[1, 2], rug = True)
    f.tight_layout()
    plt.subplots_adjust(top = 0.9)
    
    return plt.show()

def island_abm_run_simulations(sample_size):
    """ 
    Run simulations of the Island model and time it. 
        
    """
    # Set the ABM parameters and support
    island_abm_exploration_range = np.array([(0.0, 10.0),  # rho            (degree of locality in the diffustion of knowledge)
                                             (0.0, 5.0),   # lambda_param   (mean of Poisson r.v. - jumps in technology)
                                             (0.8, 2.0),   # alpha          (productivity of labour in extraction)
                                             (0.0, 1.0),   # phi            (cumulative learning effect)
                                             (0.0, 1.0),   # pi             (probability of finding a new island)
                                             (0.0, 1.0)])  # eps           (willingness to explore)
    
    # Get parameter sets
    X_train, X_test = get_unirand_parameter_samples(n_train_samples, 0, island_abm_exploration_range)
    
    # Run the ABM on the train and test parameters in BINARY case...
    start_time_binary = timeit.timeit()
    y_train, y_test =  island_abm_evaluate_samples(X_train, X_test, "binary")
    elapsed_time_binary = timeit.timeit() - start_time
    
    # Make dataframe of the training data
    df_binary = pd.concat([pd.DataFrame(X_train, columns = ["rho", "lambda_param", "alpha", "phi", "pi", "eps"]), 
                           pd.DataFrame(y_train, columns = ["result"])], 
                           axis = 1)

    # Plot the prior distributions of each parameter
    plot_prior_binary = island_abm_make_plot(df_binary, 
                                             "(Binary) Prior distributions for " + str(df_binary.shape[0]) + " simulations")

    # Subset the dataframe to just the accepted parameters
    df_binary_success = df_binary[df_binary["result"] == 1.0]

    # Plot the posterior distributions of each parameter
    plot_posterior_binary = island_abm_make_plot(df_binary_success, 
                                                 "(Binary) Posterior distributions for " + str(df_binary_success.shape[0]) + " acceptances")

    # Run the ABM on the train and test parameters in REALVALUED case...
    start_time_realvalued = timeit.timeit()
    y_train, y_test =  island_abm_evaluate_samples(X_train, X_test, "real_valued")
    elapsed_time_realvalued = timeit.timeit() - start_time_realvalued

    # Make dataframe of the training data
    df_realvalued = pd.concat([pd.DataFrame(X_train, columns = ["rho", "lambda_param", "alpha", "phi", "pi", "eps"]), 
                               pd.DataFrame(y_train, columns = ["result"])], 
                               axis = 1)

    # Plot the prior distributions of each parameter
    plot_prior_realvalued = island_abm_make_plot(df_realvalued, 
                                                 "(Real Valued) Prior distributions for " + str(df_realvalued.shape[0]) + " simulations")

    # Subset the dataframe to just the accepted parameters, this is where we define eps
    eps = 1
    df_realvalued_success = df_realvalued[df_realvalued["result"] > eps]
 
    # Plot the posterior distributions of each parameter
    plot_posterior_realvalued = island_abm_make_plot(df_realvalued_success, 
                                                     "(Real Valued) Posterior distributions for " + str(df_realvalued_success.shape[0]) + " acceptances (eps = " + str(eps) + ")")
    
    return elapsed_time_binary, elapsed_time_realvalued, plot_prior_binary, plot_prior_realvalued, plot_posterior_binary, plot_posterior_realvalued

def island_run_range_simulations(samples_sizes):
    """ 
    Run simulations of the Island model and time multiple. 
    """    
    results = []
    
    for sample_size in sample_sizes:
        result = island_abm_run_simulations(sample_size)
        results.append([sample_size, result])
    
    return results
    
        
    