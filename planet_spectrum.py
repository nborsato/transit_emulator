def read_kitz_template(file_name):
    from astropy.io import fits
    import numpy as np
    file_path = "example_spectra/templates/" + file_name
    data = fits.getdata(file_path)
    # reverse the order of the date because the kitzmann templates have the wavelength axis in reverse
    return np.array([data[0][::-1], data[1][::-1]])


def slice_spectrum(wavelength, flux, min_wavelength, max_wavelength):
    """
    Slice a given spectrum to fit within a specific wavelength range.

    Parameters:
    - wavelength_row (numpy array): The row array of wavelengths (in nanometres).
    - flux_row (numpy array): The row array of flux values corresponding to the wavelengths.
    - min_wavelength (float): The minimum wavelength of the desired range (in nanometres).
    - max_wavelength (float): The maximum wavelength of the desired range (in nanometres).

    Returns:
    - sliced_wavelength (numpy array): The sliced row array of wavelengths.
    - sliced_flux (numpy array): The sliced row array of flux values.
    """
    import numpy as np  # Importing NumPy within the function

    # Find the indices where the wavelength is within the desired range
    indices = np.where((wavelength >= min_wavelength) & (wavelength <= max_wavelength))

    # Slice the wavelength and flux rows using these indices
    sliced_wavelength = wavelength[indices]
    sliced_flux = flux[indices]

    return sliced_wavelength, sliced_flux


def reduce_resolution(wave, flux, target_res):
    """Reduce and resample the resolution of a spectrum to a given R."""

    import numpy as np
    from scipy.signal import convolve
    from scipy.interpolate import interp1d

    # Calculate the mean wavelength for determining FWHM and sigma
    lambda_median = np.median(wave)

    # Calculate FWHM and sigma
    fwhm = lambda_median / target_res
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Create Gaussian kernel
    num_points = max(10 * int(sigma), 50)  # Ensure at least 50 points
    x = np.linspace(-5 * sigma, 5 * sigma, num_points)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))

    # Normalize kernel
    kernel /= np.sum(kernel)

    # Perform convolution to reduce resolution
    reduced_flux = convolve(flux, kernel, mode='same')

    # Create a new wavelength grid based on the target resolution
    wave_new = np.arange(wave[0], wave[-1], fwhm)

    # Interpolate the reduced flux onto the new wavelength grid
    interp_function = interp1d(wave, reduced_flux, kind='linear', fill_value='extrapolate')
    resampled_flux = interp_function(wave_new)

    return wave_new, resampled_flux


def rv_broadening(wavelengths, flux, fwhm_kms):
    """
    Apply broadening to a spectrum using a Gaussian kernel based on FWHM in km/s.

    Parameters:
    wavelengths (array-like): Wavelengths of the spectrum in the same units (e.g., nm or Angstrom)
    flux (array-like): Flux values of the spectrum
    fwhm_kms (float): Full Width at Half Maximum in km/s

    Returns:
    array: Broadened flux values of the spectrum
    """

    # Import necessary modules
    from maths_functions import doppler_shift  # Custom function for Doppler shift
    import numpy as np  # For numerical operations
    from scipy.signal import convolve  # For applying the Gaussian kernel

    # Calculate the Doppler-shifted wavelength based on the FWHM in km/s
    delta_lambda = doppler_shift(np.median(wavelengths), fwhm_kms)

    # Convert FWHM from wavelength units to data points
    # Using the median wavelength for the conversion
    n_points = np.abs(int(np.round(np.median(wavelengths) / delta_lambda)))

    # Create the Gaussian filter kernel
    # Convert FWHM to sigma for Gaussian kernel
    sigma = n_points / (2 * np.sqrt(2 * np.log(2)))
    x = np.linspace(-n_points, n_points, 2 * n_points + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # Normalize the kernel

    # Apply the Gaussian filter to the spectrum
    broadened_flux = convolve(flux, kernel, mode='same')

    return broadened_flux


def interpolate_planet_spectrum(planet_wavelengths, planet_flux, stellar_wavelengths):
    """
    Interpolate the flux values onto a new wavelength grid.

    Parameters:
    known_wavelengths (array-like): Known wavelengths in the same units (e.g., nm or Angstrom)
    known_flux (array-like): Known flux values
    target_wavelengths (array-like): Target wavelengths for interpolation

    Returns:
    array: Interpolated flux values at target wavelengths
    """
    from scipy.interpolate import interp1d

    # Create an interpolation function based on the known wavelengths and flux
    interp_function = interp1d(planet_wavelengths, planet_flux, kind='linear', bounds_error=False, fill_value=0)

    # Use the interpolation function to find the flux values at the target wavelengths
    target_flux = interp_function(stellar_wavelengths)

    return target_flux
