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
wendel_min = params["wendel_min"]  # spectrographs minimum wavelength range
wendel_max = params["wendel_max"]  # spectrographs maximum wavelength range
trim_val = params["trim_val"]  # how many pixels you want to trim off the CCF function to reduce edge effects
number_transit = params["number_transit"]  # number of transits to simulate
mag_vals = params["mag_vals"]  # Magnitudes you want to simulate
number_of_cores = params["cores"]

# Read the initial spectrum template from a FITS file
wave, flux = read_kitz_template(f"{template_name}.fits")

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

# Remove the last radial velocity and phase angle value to match the differences
radial_velocities = radial_velocities[:-1]
phase_angles = phase_angles[:-1]

# Introduce the stellar model
stellar_wavelength_grid = generate_wavelength_grid(wendel_min, wendel_max, resolution)

# Interpolate planet spectrum to stellar grid model
flux = interpolate_planet_spectrum(wave, flux, stellar_wavelength_grid)
wave = stellar_wavelength_grid


def simulate_transite(j, flux, wave, exposures, v_mag, trim_val):

    # Import necessary modules for plotting, numerical operations, and custom functions
    import numpy as np
    from planet_spectrum import read_kitz_template  # Custom functions for spectrum manipulation
    from planet_spectrum import rv_broadening
    from maths_functions import doppler_shift  # Custom function for resolution calculation
    from cross_correlate import apply_cross_correlation, kp_vsys_diagram, trim_array

    # Initialize lists to store the shifted spectra
    waves = []
    fluxes = []
    stellar_noise_spectrum = np.load(f'noise_arrays/m{v_mag}/' + f'noise_{j}.npy')

    # Apply radial velocity broadening to the spectrum, and also shift lines to correct position and store the results
    # We also combine the stellar model and the planet model together through summation
    for i in range(0, exposures):
        # ensure that the random seed changes for each loop
        flux_rv = rv_broadening(wave, flux, radial_velocities_diff[i])

        # calculate_snr_based_on_v_mag(v_mag)
        flux_rv = flux_rv + stellar_noise_spectrum[i] + 1
        wave_shift = doppler_shift(wave, radial_velocities[i])
        waves.append(wave_shift)
        fluxes.append(flux_rv)

    # Load the spectral template for cross-correlation.
    template = read_kitz_template(f"{template_name}.fits")

    # Apply cross-correlation to the list of shifted spectra.
    ccf, rv_grid = apply_cross_correlation(waves, fluxes, rv_bounds, template)

    # Create a Kp-Vsys diagram based on the loaded cross-correlation function.
    kp_vsys = kp_vsys_diagram(ccf, phase_angles, np.linspace(kpvsys_min, kpvsys_max, kpvsys_max),
                              velocity_semiamplitude)


    return [rv_grid, ccf, kp_vsys]


exposures = len(phase_angles)
for mag_val in mag_vals:

    save_noise_arrays(number_transit, wave, phase_angles, mag_val)
    print(f"Simulating v_mag: {mag_val}")
    par_list = [[i, flux, wave, exposures, mag_val, trim_val] for i in range(1, number_transit+1)]

    ccf_results = Parallel(n_jobs=number_of_cores)(
        delayed(simulate_transite)(j, flux, wave, exposures,mag_val,trim_val) for j, flux, wave,exposures,mag_val,
        trim_val in par_list)

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

