def calculate_ccf(wld, fxd, wl, fx2,rv_range):
    """
    Cross-correlate a given spectrum with a template.

    Parameters:
    wld (array-like): Clipped wavelength range for the spectrograph to avoid edge effects.
    fxd (array-like): Corresponding clipped flux range for the spectrograph.
    wl (array-like): Total wavelength range of the template.
    fx2 (array-like): Flux values of the template.
    rv_range (value): Specifiy the symmetrical RV bounds of the cross-correlation search

    Returns:
    array: Radial velocity array.
    array: Cross-correlation function.
    """
    import tayph.ccf as ccf
    import numpy as np
    from contextlib import redirect_stdout, redirect_stderr
    from io import StringIO

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        rv, cc_function, _ = ccf.xcor([wld], [np.vstack((fxd, fxd))], [wl], [fx2], 1.0, rv_range)

    return rv, cc_function[0][0]


def apply_cross_correlation(waves, fluxes, rv_bounds, template):
    """
    Apply cross-correlation to a list of spectra using a template spectrum.

    Parameters:
    transit_info (list of arrays): List containing [wavelength, flux] arrays for each exposure.
    template (array-like): An array containing [template_wavelength, template_flux].

    Returns:
    list: A list of cross-correlated flux values.
    array: Radial velocity array corresponding to the last exposure.
    """
    import os
    import numpy as np
    from tqdm import tqdm
    import tayph.ccf as ccf

    ccs = []  # List to store cross-correlated flux values

    print("Applying Cross Correlation")

    def calculate_cc(wld, fxd, wl, fx2):
        """
        Cross-correlate a given spectrum with a template.

        Parameters:
        wld (array-like): Clipped wavelength range for the spectrograph to avoid edge effects.
        fxd (array-like): Corresponding clipped flux range for the spectrograph.
        wl (array-like): Total wavelength range of the template.
        fx2 (array-like): Flux values of the template.

        Returns:
        array: Radial velocity array.
        array: Cross-correlation function.
        """
        # Redirect stdout and stderr
        fd = os.open(os.devnull, os.O_WRONLY)
        old_fd_stdout = os.dup(1)
        old_fd_stderr = os.dup(2)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)

        # Perform the cross-correlation
        rv, cc_function, _ = ccf.xcor([wld], [np.vstack((fxd, fxd))], [wl], [fx2], 1.0, rv_bounds)

        # Revert redirection of stdout and stderr
        os.dup2(old_fd_stdout, 1)
        os.dup2(old_fd_stderr, 2)

        return rv, cc_function[0][0]

    # Loop through each exposure and apply cross-correlation
    for i in tqdm(range(len(waves))):
        cc = calculate_cc(waves[i], fluxes[i], template[0], template[1])  # Apply cross-correlation
        ccs.append(cc[1])  # Store cross-correlated flux values

    ccs = np.array(ccs) - np.median(ccs)

    return ccs, cc[0]  # Return cross-correlation function and radial velocity grid


def kp_vsys_diagram(array, phases, kp_values, velocity_semi_amplitude):
    """
    Create a Kp-Vsys diagram by shifting CCFs based on a range of Kp values and orbital phases.

    Parameters:
    array (numpy.ndarray): A 2D array of cross-correlation values.
    phases (list): A list of different phases for the rows in the array.
    kp_values (list): A list of different Kp values to test.
    velocity_semi_amplitude (float): The known velocity semi-amplitude value for initial shift.
    template_name (str): The name of the template used, for file naming.

    Returns:
    numpy.ndarray: A 2D array representing the Kp-Vsys diagram.
    """
    import numpy as np

    def shift_rows(array, phases, kp):
        """
        Shift the rows of a 2D array based on phases and a given Kp value.

        Parameters:
        array (numpy.ndarray): The array to shift.
        phases (list): The phases corresponding to each row in the array.
        kp (float): The Kp value to use for shifting.

        Returns:
        numpy.ndarray: The shifted array.
        """
        shifted_array = np.copy(array)
        for i, phase in enumerate(phases):
            v_t = kp * np.sin(2 * np.pi * phase)  # Calculate apparent radial velocity
            radial_velocities = np.arange(array.shape[1])
            shifted_radial_velocities = radial_velocities + v_t  # Shift radial velocities
            shifted_values = np.interp(radial_velocities, shifted_radial_velocities, shifted_array[i])
            shifted_array[i] = shifted_values  # Update shifted values

        return shifted_array

    kp_vsys = np.zeros((len(kp_values), array.shape[1]))
    kp_vys_array = []

    # Initial shift using known semi-amplitude value
    array = shift_rows(array, phases, -velocity_semi_amplitude)

    for k, kp in enumerate(kp_values):
        shifted_array = shift_rows(array, phases, kp)
        kp_vsys[k] = np.sum(shifted_array, axis=0)  # Co-add signal for the current Kp value
        kp_vys_array.append(shifted_array)

    return kp_vsys


def extract_1d_spectrum_from_ccf(rv, kpvsys_array, v_orb):
    """
    Extract a 1D spectrum from the Kp-Vsys array.

    Parameters:
    kpvsys_array (numpy.ndarray): A 2D array representing the Kp-Vsys diagram.
    v_orb (float): The orbital velocity.
    rv (array): The radial velocity array.
    template_name (str): The name of the template used, for file naming.

    Returns:
    numpy.ndarray: A 1D array containing the extracted spectrum.
    """
    import numpy as np

    # Extract the 1D CCF corresponding to the orbital velocity
    one_d_ccf = kpvsys_array[np.abs(int(v_orb))]

    # Create an array containing radial velocity and 1D CCF
    one_d_info = np.array([rv, one_d_ccf])

    return one_d_info


def trim_array(arr, trim_value):
    """
    Trims an array by removing 'trim_value' number of elements
    from both the beginning and end of each dimension.

    Parameters:
    - arr (array-like): The array to trim.
    - trim_value (int): The number of elements to remove from both ends.

    Returns:
    - array: The trimmed array.
    """
    import numpy as np

    # Transpose for trimming
    arr_transposed = arr.T

    # Trim along the first dimension
    arr_trimmed_1 = arr_transposed[trim_value:-trim_value]

    # Transpose back and trim along the other dimension
    arr_trimmed_2 = arr_trimmed_1.T[trim_value:-trim_value]

    return arr_trimmed_2.T
