import numpy as np

def get_unirand_parameter_samples(n_train_samples, n_test_samples, parameter_support):
    """
    Generate n_samples from the parameter support
    """
    # Get the range for the support, array containing the range of each parameter
    support_range = parameter_support[:, 1] - parameter_support[:, 0]
    
    # Generate training sample
    random_samples = np.random.rand(parameter_support.shape[0], n_train_samples+ n_test_samples).T
    unirand_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])
    
    unirand_train_samples = unirand_samples[0: n_train_samples, ]
    unirand_test_samples = unirand_samples[n_train_samples:, ]

    return unirand_train_samples, unirand_test_samples

print "get_unirand_parameter_samples successfully imported"

print "import generic_abm complete"
