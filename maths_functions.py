def gaussian(x, sigma):
    """Return a Gaussian function."""
    import numpy as np
    return np.exp(-x ** 2 / (2 * sigma ** 2))


def calculate_resolution(wavelength_row):
    """
    Calculate the spectral resolution for a given digital spectrum.

    Parameters:
    - wavelength_row (numpy array): The row array of wavelengths (in nanometres).

    Returns:
    - resolution (numpy array): The array of resolutions at each wavelength point.
    """

    import numpy as np

    # Calculate the differences between adjacent wavelength points
    delta_lambda = np.median(np.diff(wavelength_row))

    # Calculate the resolution
    resolution = np.median(wavelength_row) / delta_lambda

    return int(resolution)


def doppler_shift(wavelengths, velocity):
    """
    Apply the Doppler shift to an array of wavelengths.

    Parameters:
    wavelengths (array-like): Original wavelengths in the same units (e.g., nm or Angstrom)
    velocity (float): Velocity of the source relative to the observer in km/s

    Returns:
    array: Doppler-shifted wavelengths
    """

    # Speed of light in km/s
    c = 299792.458

    # Apply the Doppler shift formula
    shifted_wavelengths = wavelengths * (1 + velocity / c)

    return shifted_wavelengths
