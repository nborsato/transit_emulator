import pdb

from statistics_constraints import fit_gaussian, obtain_detection_sig, av_kpvsys, bin_down_data, bin_down_data_2d
import matplotlib.pyplot as plt
import os
import numpy as np
import json

# Read parameters from JSON file
with open('0_parameters.json', 'r') as file:
    params = json.load(file)

template_name = params["template_name"]
mag_vals = params["mag_vals"]

def write_to_file(results, output_path):
    # Create a text file to store the detection significances
    with open(os.path.join(output_path, 'detection_significances.txt'), 'w') as f:
        for num, sig in results.items():
            f.write(f"Detection significance for {num} transits: {sig}\n")

def analyze_transits(num_transits, kpvsyss, rv, plot=False):
    # Compute averaged kp-vsys array for the given number of transits
    kp_n_transit = av_kpvsys(kpvsyss, num_transits, rv)
    # Calculate the detection significance for the averaged kp-vsys array
    n_transit_sig = obtain_detection_sig(kp_n_transit, rv,bootstrap_iterations=1000)

    # Bin down the radial velocity data and fit a Gaussian to it
    binned_rv = bin_down_data(rv)
    binned_ccf = bin_down_data(kp_n_transit[198])  # Assuming the orbital velocity index is 234
    rv_grid, y_data, fitted_data, params, diag_matrix = fit_gaussian(binned_ccf, binned_rv)

    # Invert the data for plotting
    y_data = y_data * -1
    fitted_data = fitted_data * -1

    if plot==True:

        # Plotting
        plt.imshow(bin_down_data_2d(kp_n_transit), origin="lower", interpolation="none")
        plt.show()

        plt.plot(rv_grid, y_data)
        plt.plot(rv_grid, fitted_data)
        #plt.xlim(-100, 100)
    plt.show()

    return n_transit_sig


for mag_value in mag_vals:

    print(f"Testing significances for {mag_value} magnitude")
    # Set the output paths for the data files
    output_path = f"ccf_output/{template_name}/m{mag_value}/"
    ccf_output = os.path.join(output_path, "ccfs.npy")
    kpvsys_output = os.path.join(output_path, "kpvsys_arrays.npy")
    rv_output = os.path.join(output_path, "rv.npy")

    # Check if the files exist before trying to load them
    if os.path.exists(ccf_output) and os.path.exists(kpvsys_output) and os.path.exists(rv_output):

        # Load data from files
        ccfs = np.load(ccf_output)
        kpvsyss = np.load(kpvsys_output)
        rv = np.load(rv_output)
        # Analyze for different numbers of transits
        transit_numbers = range(1, len(kpvsyss) + 1)
        results = {}

        for num in transit_numbers:
            sig = analyze_transits(num, kpvsyss, rv, plot=False)
            results[num] = sig
            print(f"Detection significance for {num} transits: {sig}")

        # Write the results to a text file in the output directory
        write_to_file(results, output_path)
    else:
        print(f"Data files for magnitude {mag_value} not found. Skipping...")
