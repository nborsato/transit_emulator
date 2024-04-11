import pdb

from joblib import Parallel, delayed
from snr_calculate import calculate_snr_based_on_v_mag  # Custom function for SNR calculation
from planet_spectrum import read_kitz_template, slice_spectrum, reduce_resolution
from planet_spectrum import interpolate_planet_spectrum
from maths_functions import calculate_resolution  # Custom function for resolution calculation
from planet_orbit import calculate_radial_velocity, calculate_orbital_velocity, calculate_phase_angle
from stellar_spectrum import generate_wavelength_grid, generate_gaussian_noise_spectrum,save_noise_arrays
import numpy as np
import json
import os

# Function to ensure directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Read parameters from JSON file
with open('0_parameters.json', 'r') as file:
    params = json.load(file)

# Assign parameters
transit_duration = params["transit_duration"]  # convert to seconds
t = transit_duration / 2  # current time in days
t0 = params["t0"]  # time at which phase is zero in days
P = params["P"]  # orbital period in days
semi_major_axis = params["semi_major_axis"]  # semi-major axis in AU
vsys = params["vsys"]  # systemic velocity in km/s
velocity_semiamplitude = params["velocity_semiamplitude"]
exposure_time = params["exposure_time"]  # the simulated exposure time of the transit
resolution = params["resolution"]  # spectrograph resolution
rv_bounds = params["rv_bounds"]  # radial velocity bounds of the CCF function
kpvsys_min = params["kpvsys_min"]  # minimum kpvsys value
kpvsys_max = params["kpvsys_max"]  # maximum kpvsys value
template_name = params["template_name"]  # name of the template used in the analysis
planet_model = params["planet_model"]
wendel_min = params["wendel_min"]  # spectrographs minimum wavelength range
wendel_max = params["wendel_max"]  # spectrographs maximum wavelength range
trim_val = params["trim_val"]  # how many pixels you want to trim off the CCF function to reduce edge effects
number_transit = params["number_transit"]  # number of transits to simulate
mag_vals = params["mag_vals"]  # Magnitudes you want to simulate
number_of_cores = params["cores"]
template_names = ["HI_m_0_t_4000"]

for template_name in template_names:
    print(template_name)
    # Read the initial spectrum template from a FITS file
    wave, flux = read_kitz_template(f"{planet_model}.fits")
    flux = flux - 1

    # Reduce the spectrum's resolution
    wave, flux = reduce_resolution(wave, flux, resolution)

    # Slice the spectrum to keep only the wavelengths between 383 and 885 nm
    wave, flux = slice_spectrum(wave, flux, wendel_min, wendel_max)

    # Calculate the orbital velocity
    v_orb = calculate_orbital_velocity(P, semi_major_axis)

    # Define steps in the transit
    transit_steps = np.arange(-t, t, exposure_time)

    # Calculate phase angles for each step in the transit
    phase_angles = calculate_phase_angle(transit_steps, t0, P * 24 * 60 * 60)

    # Calculate radial velocities at each transit step
    radial_velocities = calculate_radial_velocity(v_orb, phase_angles, vsys)

    # Calculate the differences between adjacent radial velocities
    radial_velocities_diff = np.diff(radial_velocities)
    #pdb.set_trace()

    # Remove the last radial velocity and phase angle value to match the differences
    radial_velocities = radial_velocities[:-1]
    #phase_angles = phase_angles[:-1]

    # Introduce the stellar model
    stellar_wavelength_grid = generate_wavelength_grid(wendel_min, wendel_max, resolution)

    # Interpolate planet spectrum to stellar grid model
    flux = interpolate_planet_spectrum(wave, flux, stellar_wavelength_grid)
    wave = stellar_wavelength_grid

    def simulate_transite(j, flux, wave, rvs, v_mag, vsys):
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        import scipy.ndimage as scint
        # Import necessary modules for plotting, numerical operations, and custom functions
        import numpy as np
        from planet_spectrum import read_kitz_template  # Custom functions for spectrum manipulation
        from planet_spectrum import rv_broadening
        from maths_functions import doppler_shift  # Custom function for resolution calculation
        from cross_correlate import apply_cross_correlation, make_kp_vsys

        # Initialize lists to store the shifted spectra
        waves = []
        fluxes = []
        stellar_noise_spectrum = np.load(f'noise_arrays/m{v_mag}/' + f'noise_{j}.npy')*0.5

        # Apply radial velocity broadening to the spectrum, and also shift lines to correct position and store the results
        # We also combine the stellar model and the planet model together through summation
        for i in range(0, len(rvs)):
            # ensure that the random seed changes for each loop
            flux = rv_broadening(wave, flux, radial_velocities_diff[i])

            #Shifting signal based on planet velocity
            wave_shift = doppler_shift(wave, rvs[i])

            #Shifting signal to stellar rest frame
            wave_shift = doppler_shift(wave_shift, vsys*-1)

            # Interpolate the original flux values onto the new, shifted wavelength grid
            # Ensure that the interpolation is done within the bounds of the original wavelength array
            flux_interpolator = interp1d(wave, flux, kind='nearest', fill_value="extrapolate")
            flux_rv_shifted = flux_interpolator(wave_shift)
            flux_rv_shifted = flux_rv_shifted + stellar_noise_spectrum[i]

            # Store the shifted wavelength and flux values
            waves.append(wave)  # Store the shifted wavelengths if needed
            fluxes.append(flux_rv_shifted)


        ccfs_dir = f'ccf_output/{template_name}/m{str(mag_val)}/'
        ensure_dir(ccfs_dir)
        np.save(ccfs_dir + 'orbital_velocities.npy', rvs)
        np.save(ccfs_dir + 'wave_grid.npy', wave)

        # Load the spectral template for cross-correlation.
        template = read_kitz_template(f"{template_name}.fits")

        # Apply cross-correlation to the list of shifted spectra.
        ccf, rv_grid = apply_cross_correlation(waves, fluxes, rv_bounds, template)

        # Create a Kp-Vsys diagram based on the loaded cross-correlation function.
        kp_vsys = make_kp_vsys(ccf, phase_angles)


        return [rv_grid, ccf, kp_vsys]


    exposures = len(phase_angles)
    for mag_val in mag_vals:

        save_noise_arrays(number_transit, wave, phase_angles, mag_val)
        print(f"Simulating v_mag: {mag_val}")
        par_list = [[i, flux, wave, radial_velocities, mag_val, vsys] for i in range(1, number_transit+1)]

        ccf_results = Parallel(n_jobs=number_of_cores)(
            delayed(simulate_transite)(j, flux, wave, radial_velocities,mag_val,vsys) for j, flux, wave, rvs,mag_val,vsys in par_list)

        ccfs = []
        kpvsyss = []
        for i in range(len(ccf_results)):
            ccf_result = ccf_results[i]
            ccf = ccf_result[1]
            ccfs.append(ccf)
            kpvsys = ccf_result[2]
            kpvsyss.append(kpvsys)

        rv = ccf_result[0]
        ccfs = np.array(ccfs)
        kpvsyss = np.array(kpvsyss)

        # Directory paths for saving files
        ccfs_dir = f'ccf_output/{template_name}/m{str(mag_val)}/'
        ensure_dir(ccfs_dir)

        # Save files
        np.save(ccfs_dir + 'ccfs.npy', ccfs)
        np.save(ccfs_dir + 'rv.npy', rv)
        np.save(ccfs_dir + 'kpvsys_arrays.npy', kpvsyss)

