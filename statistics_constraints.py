import pdb


def gaussian(x, a, b, c):
    """
    Gaussian function for curve fitting.

    Parameters:
    x (numpy.ndarray): The independent variable where the data is measured.
    a (float): Amplitude of the Gaussian.
    b (float): Mean of the Gaussian.
    c (float): Standard deviation of the Gaussian.

    Returns:
    numpy.ndarray: The Gaussian values evaluated at each point in x.
    """
    import numpy as np
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def process_data(path_to_data, night, filename, factor=1):
    """
    Process the FITS data.

    Parameters:
    path_to_data (str): Directory path to the data.
    night (str): The night the data was taken.
    filename (str): The name of the FITS file.
    factor (float): The scaling factor for the data.

    Returns:
    numpy.ndarray: The processed data.
    """
    from astropy.io import fits
    data = fits.getdata(path_to_data + night + filename)
    data = data.T
    data = data * factor * -1e6
    data = data.T
    return data


def remove_continuum(y_data, rv_grid, rv_low, rv_high, poly_degree):
    """
    Remove the continuum from the given data using polynomial fitting.

    Parameters:
    y_data (numpy.ndarray): The data from which the continuum needs to be removed.
    rv_grid (numpy.ndarray): The radial velocity grid corresponding to y_data.
    rv_low (float): The lower bound for excluding data points for polynomial fitting.
    rv_high (float): The upper bound for excluding data points for polynomial fitting.
    poly_degree (int): The degree of the polynomial used for fitting.

    Returns:
    numpy.ndarray: The data with the continuum removed.
    """
    import numpy as np

    # Define bounds for regions to exclude in the polynomial fit
    exclude_low = rv_low
    exclude_high = rv_high

    # Create a mask to exclude data points between rv_low and rv_high
    mask = (rv_grid < exclude_low) | (rv_grid > exclude_high)

    # Apply the mask to get the regions used for polynomial fitting
    data_masked = y_data[mask]
    rv_grid_masked = rv_grid[mask]

    # Perform polynomial fitting to estimate the continuum
    cont_polynomial = np.poly1d(np.polyfit(rv_grid_masked, data_masked, poly_degree))

    # Subtract the fitted polynomial from the original data to remove the continuum
    y_data = y_data - cont_polynomial(rv_grid)

    return y_data


def fit_gaussian(data, rv_grid):
    """
    Fit a Gaussian function to a given set of data.

    Parameters:
    data (numpy.ndarray): The data array to which the Gaussian is to be fitted.
    index (int): Index specifying which subset of the data array to fit.
    rv_bounds (int): The bounds for the radial velocity grid.

    Returns:
    tuple: Contains the radial velocity grid, the original and fitted data,
           the parameters of the fitted Gaussian, and their standard deviations.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    # Extract the specific subset of data to be fitted
    y_data = data

    # Remove the continuum from the data before fitting
    y_data = remove_continuum(y_data, rv_grid, -30, 0, 8)

    # Initial parameter guesses for Gaussian fitting: [amplitude, mean, standard deviation]
    init_params = [-0.03, -20, 5]

    # Perform Gaussian fitting
    params, params_cov = curve_fit(gaussian, rv_grid, y_data, p0=init_params)

    # Generate the Gaussian fitted data
    fitted_data = gaussian(rv_grid, *params)

    # Extract the standard deviations of the fitted parameters
    std_devs = np.sqrt(np.diag(params_cov))

    return rv_grid, y_data, fitted_data, params, std_devs


def perform_noise_injection_resampling(rv, data, num_resamples, model_peak):
    """
    Perform noise injection resampling to estimate uncertainties.

    Parameters:
    rv (numpy.ndarray): Radial velocity array.
    data (numpy.ndarray): The data array.
    num_resamples (int): The number of resampling iterations.
    model_peak (float): The model peak value for discrepancy calculation.

    Returns:
    numpy.ndarray: Resampled parameters and their deviations from the model.
    """
    # Import necessary modules
    import numpy as np
    from scipy.optimize import curve_fit
    from tqdm import tqdm
    from scipy.stats import median_abs_deviation
    import warnings
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)

    # Initialize empty lists to store resampled parameters and deviations
    resampled_params = []
    deviations = []

    # Initial guesses for the Gaussian parameters
    init_params = [250, -18, 20]

    # Define the range of radial velocities to exclude from the continuum fitting
    exclude_low = -30
    exclude_high = 0
    poly_deg = 8

    # Create a mask to exclude certain radial velocities
    mask = (rv < exclude_low) | (rv > exclude_high)
    data_masked = data[mask]

    # Remove continuum variations as this will affect the fit
    data = remove_continuum(data, rv, exclude_low, exclude_high, poly_deg)

    # Reduce number of correlated values
    step_skip = 4
    rv = rv[::step_skip]
    data = data[::step_skip]

    # Calculate the standard deviation of the masked data for noise injection
    mad_masked = median_abs_deviation(data_masked)
    std_masked = 1.4826 * mad_masked  # scale MAD to get the standard deviation under normal assumption

    # Loop over the number of resamples
    for _ in tqdm(range(num_resamples), desc="Resampling"):
        # Generate random noise based on the standard deviation
        noise = np.random.normal(0, std_masked, len(data))

        # Add the random noise to the data
        resampled_data = data + noise

        # Sort data by radial velocity for better fitting
        sort_indices = np.argsort(rv)
        resampled_data = resampled_data[sort_indices]
        resampled_rv = rv[sort_indices]

        try:
            # Fit a Gaussian to the resampled data
            params, _ = curve_fit(gaussian, resampled_rv, resampled_data, p0=init_params, maxfev=10000)

            # Store the fitted parameters
            resampled_params.append(params)

            # Calculate the discrepancy between the model and the fitted peak
            model_discrepancy = params[0] / model_peak

            # Store the discrepancy
            deviations.append(model_discrepancy)

        except RuntimeError:
            continue

    return np.array(resampled_params), np.array(deviations)


def obtain_detection_sig(kpvsys, rv, bootstrap_iterations=1000, ccf_amplitude=-0.003, v_orb=234):
    """
    Obtain the detection significance based on the kp-vsys array and radial velocity grid.

    Parameters:
    kpvsys (numpy.ndarray): 2D array representing the kp-vsys diagram.
    rv (numpy.ndarray): 1D array of radial velocities.
    bootstrap_iterations (int, optional): Number of bootstrap iterations for noise injection resampling. Default: 1000.
    ccf_amplitude (float, optional): First guess for the amplitude of the CCF function. Default: -0.003.
    v_orb (int, optional): Orbital velocity index. Default: 234.

    Returns:
    float: Detection significance.
    """
    from cross_correlate import extract_1d_spectrum_from_ccf
    import numpy as np

    # Extract a 1D cross-correlation function from the kp-vsys diagram.
    one_d_ccf = extract_1d_spectrum_from_ccf(rv, kpvsys, v_orb)

    # Bin down the data to reduce the noise
    binned_rv = bin_down_data(one_d_ccf[0])
    binned_ccf = bin_down_data(one_d_ccf[1])

    # Fit a Gaussian to the binned data to locate the position and parameters of the peak
    rv_grid, y_data, fitted_data, params, diag_matrix = fit_gaussian(binned_ccf, binned_rv)

    # Perform noise injection resampling to estimate the detection significance
    bootstrap_params, bootstrap_discr = perform_noise_injection_resampling(one_d_ccf[0], one_d_ccf[1],
                                                                           bootstrap_iterations, ccf_amplitude)

    # Calculate the standard deviation of the bootstrap parameters
    std_bootsrap = np.std(bootstrap_params, axis=0)

    # Calculate the detection significance
    detection_sig = np.round(np.absolute(params[0] / std_bootsrap[0]), 1)

    return detection_sig


def av_kpvsys(kpvsyss, obs_number, rv, lower_bound=-30, upper_bound=0, smoothing_factor=8):
    """
    Compute the average kp-vsys array for a given number of observations and apply continuum removal.

    Parameters:
    kpvsyss (numpy.ndarray): 3D array containing kp-vsys arrays for multiple observations.
    obs_number (int): Number of observations to consider for averaging.
    rv (numpy.ndarray): 1D array of radial velocities.
    lower_bound (int, optional): Lower bound for continuum removal. Defaults to -30.
    upper_bound (int, optional): Upper bound for continuum removal. Defaults to 0.
    smoothing_factor (int, optional): Smoothing factor for continuum removal. Defaults to 8.

    Returns:
    numpy.ndarray: Averaged kp-vsys array with continuum removed.
    """
    import numpy as np

    # Select the kp-vsys arrays corresponding to the number of observations
    kpvsys_obs = kpvsyss[:obs_number]

    # Compute the mean kp-vsys array
    mean_kpvsys = np.mean(kpvsys_obs, axis=0)

    # Remove the continuum from each row in the mean kp-vsys array
    for i in range(len(mean_kpvsys)):
        mean_kpvsys[i] = remove_continuum(mean_kpvsys[i], rv, lower_bound, upper_bound, smoothing_factor)

    return mean_kpvsys


def bin_down_data(data, window_size=4):
    """
    Bin down data by averaging over the nearest neighbours.

    Parameters:
    data (numpy.ndarray): The data array to bin down.
    window_size (int): The size of the moving window (default is 5).

    Returns:
    numpy.ndarray: The binned data array.
    """
    import numpy as np
    # Check if the window_size is odd, if not increase by 1
    if window_size % 2 == 0:
        window_size += 1

    # Calculate the number of neighbours on each side of a point
    num_neighbours = window_size // 2

    # Initialize the binned data array
    binned_data = np.zeros(len(data))

    # Loop through the data array
    for i in range(len(data)):
        # Find the indices of the neighbours
        start_idx = max(0, i - num_neighbours)
        end_idx = min(len(data), i + num_neighbours + 1)

        # Average over the neighbours
        binned_data[i] = np.mean(data[start_idx:end_idx])

    return binned_data


def bin_down_data_2d(data, window_size=5):
    """
    Bin down 2D data by averaging over the nearest neighbours along the wavelength axis.

    Parameters:
    data (numpy.ndarray): The 2D data array to bin down. Shape should be (num_wavelengths, num_measurements)
    window_size (int): The size of the moving window (default is 5).

    Returns:
    numpy.ndarray: The binned data array.
    """
    import numpy as np

    # Check if the window_size is odd, if not increase by 1
    if window_size % 2 == 0:
        window_size += 1

    # Calculate the number of neighbours on each side of a point
    num_neighbours = window_size // 2

    # Initialize the binned data array
    binned_data = np.zeros(data.shape)

    # Loop through the wavelength axis (axis=0)
    for i in range(data.shape[0]):
        # Find the indices of the neighbours
        start_idx = max(0, i - num_neighbours)
        end_idx = min(data.shape[0], i + num_neighbours + 1)

        # Average over the neighbours for each measurement and store in the binned_data array
        binned_data[i, :] = np.mean(data[start_idx:end_idx, :], axis=0)

    return binned_data