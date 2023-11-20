from astropy.table import Table
import matplotlib.pyplot as plt
from labellines import labelLines
import matplotlib as mpl
import warnings
import json

# Read parameters from JSON file
with open('0_parameters.json', 'r') as file:
    params = json.load(file)

template_name = params["template_name"]
mag_vals = params["mag_vals"]

# Ignore UserWarning from labellines.core
warnings.filterwarnings("ignore", category=UserWarning, module="labellines.core")

# Set the font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14  # Good starting point, might need to adjust.

# Colour codes
colour_codes = ['#00afaa', '#5300af', '#af0005', '#5caf00', '#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845']

dataset = f"ccf_output/{template_name}/detection_trends.csv"

# Read data
t = Table.read(dataset)
t = t[(t['Transit_Number']<30)]

# Create a new figure and axis for each plot
fig, ax = plt.subplots(figsize=(12, 6))

plt.axhline(y=3, color='k', linestyle=':', alpha=0.2)
plt.axhline(y=5, color='k', linestyle='-.', alpha=0.2)
plt.axhline(y=10, color='k', linestyle='--', alpha=0.2)
count = 0
transit_no = t['Transit_Number']

for mag_val in mag_vals:
    ax.plot(transit_no, t[f'm{mag_val}'], label=f"M({mag_val})", color=colour_codes[count])
    if count == len(colour_codes) - 1:
        count = 0
    else:
        count += 1

ax.set_ylabel(r'Detection significance ($\sigma$)')
ax.set_xlabel('Number of transits')

# Use labellines to place labels
labelLines(ax.get_lines(), zorder=2.5, fontsize=10)

# Set only bottom and left borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
save_path = f"plots/transit_significances_trend_{template_name}.pdf"
plt.savefig(save_path)  # Save file with identifier

#plt.show()