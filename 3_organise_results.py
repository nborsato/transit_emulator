import os
import pdb

from astropy.table import Table
import numpy as np
import json

# Read parameters from JSON file
with open('0_parameters.json', 'r') as file:
    params = json.load(file)

template_name = params["template_name"]
mag_vals = params["mag_vals"]

# Initialize empty lists to store data
m_values = []
transit_numbers = None  # We'll initialize this once we read the first file
detection_significances = []

# Root directory containing mX.X folders
root_dir = f'ccf_output/{template_name}'


# Loop over each directory starting with 'm'
for mag_val in mag_vals:

    file_path = os.path.join(root_dir, f"m{mag_val}", 'detection_significances.txt')

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Initialize transit_numbers list if it's None
        if transit_numbers is None:
            transit_numbers = [int(line.split(':')[0].split()[-2]) for line in lines]

        # Extract detection significance values
        current_detections = [float(line.split(':')[1].strip()) for line in lines]
        detection_significances.append(current_detections)

# Convert lists to numpy arrays for easier manipulation
m_values = np.array(m_values)
detection_significances = np.array(detection_significances)

# Create an Astropy table and fill it with data
t = Table()
t['Transit_Number'] = transit_numbers

for i in range(0,len(mag_vals)):
    t[f'm{mag_vals[i]}'] = detection_significances[i]

# Save the table as a CSV file
t.write(os.path.join(root_dir, 'detection_trends.csv'), format='csv',overwrite=True)

print("Astropy table saved as 'detection_trends.csv'.")
