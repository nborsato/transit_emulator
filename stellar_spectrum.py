def generate_gaussian_noise_spectrum(wavelengths, snr, seed_nos, num_exposures):
    """
    Generate Gaussian noise for given SNR and multiple exposures.

    Parameters:
    wavelengths (array-like): Wavelengths in the same units (e.g., nm or Angstrom)
    snr (float): Signal-to-Noise Ratio
    seed_nos (array-like): Array of seed numbers for random number generation, one for each exposure.
    num_exposures (int): Number of exposures to generate noise for.

    Returns:
    2D array: Noise spectra for each exposure
    """
    import numpy as np

    #snr = snr*(2/3)#SNR shows peak of the median blaze assume triangle and scale by 2/3 factor

    # Calculate the standard deviation of the noise
    sigma = 1 / snr

    seed_nos = seed_nos

    # Create a random number generator object with a different seed for each exposure
    rng = np.random.default_rng(seed=seed_nos)

    # Generate Gaussian noise for all exposures at once
    noise_spectra = rng.normal(0, sigma, (num_exposures, len(wavelengths)))

    return noise_spectra


def generate_poisson_noise_spectrum(wavelengths, signal, snr):
    """
    Generate Poisson noise for a given signal and SNR.
    #NOTE THIS DOESN"T WORK#

    Parameters:
    wavelengths (array-like): Wavelengths in the same units (e.g., nm or Angstrom)
    signal (array-like): Signal values (e.g., flux)
    snr (float): Signal-to-Noise Ratio

    Returns:
    array: Signal with added Poisson noise
    """
    import numpy as np

    snr = snr

    signal = (signal + snr) * snr
    # Calculate the noise level based on the signal and SNR
    noise_level = np.sqrt(signal) / snr

    # Generate Poisson noise
    noise = np.random.poisson(noise_level, size=len(wavelengths))

    # Add the noise to the original signal to get the noisy spectrum
    noisy_spectrum = (noise - snr) / snr

    return noisy_spectrum

def generate_wavelength_grid(start_wavelength, end_wavelength, resolution):
    """
    Generate a wavelength grid with a given spectral resolution.

    Parameters:
    start_wavelength (float): Start wavelength in the same units (e.g., nm or Angstrom)
    end_wavelength (float): End wavelength in the same units
    resolution (float): Desired spectral resolution (R)

    Returns:
    array: Wavelength grid
    """
    import numpy as np

    # Create an empty list to store the wavelengths
    wavelength_grid = []

    # Start with the initial wavelength
    current_wavelength = start_wavelength

    # Generate the grid
    while current_wavelength <= end_wavelength:
        wavelength_grid.append(current_wavelength)

        # Calculate the wavelength step based on the current wavelength and desired resolution
        delta_lambda = current_wavelength / resolution

        # Move to the next wavelength
        current_wavelength += delta_lambda

    return np.array(wavelength_grid)

def save_noise_arrays(number_transit, wave, phase_angles, v_mag):
    import os
    import numpy as np
    from snr_calculate import calculate_snr_based_on_v_mag


    # Define the folder name
    folder_name = f"noise_arrays/m{v_mag}"

    # Check if the folder exists; if not, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for noise_i in range(1, number_transit + 1):
        noise_spectrum = generate_gaussian_noise_spectrum(
            wave,
            calculate_snr_based_on_v_mag(v_mag),
            np.arange(len(phase_angles)) * noise_i,
            len(phase_angles)
        )

        # Save the file into the specified folder
        np.save(f'{folder_name}/noise_{noise_i}.npy', noise_spectrum)
