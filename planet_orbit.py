def calculate_orbital_velocity(period, semi_major_axis, period_unit="days", axis_unit="AU"):
    """
    Calculate the orbital velocity of a planet given its period and semi-major axis.

    Parameters:
    period -- float, the orbital period
    semi_major_axis -- float, the semi-major axis of the orbit
    period_unit -- str, unit of the period ("days" or "s"), default is "days"
    axis_unit -- str, unit of the semi-major axis ("AU" or "m"), default is "AU"

    Returns:
    velocity -- float, the orbital velocity in km/s
    """
    import numpy as np

    # Conversion factor from AU to meters
    au_to_m = 1.496e11  # 1 AU = 1.496 x 10^11 meters

    # Conversion factor from days to seconds
    days_to_s = 24 * 3600  # 1 day = 24 hours x 3600 seconds/hour

    # Convert period to seconds if needed
    if period_unit == "days":
        period = period * days_to_s

    # Convert semi-major axis to meters if needed
    if axis_unit == "AU":
        semi_major_axis = semi_major_axis * au_to_m

    # Calculate the orbital velocity
    velocity = (2 * np.pi * semi_major_axis) / period  # in m/s

    # Convert to km/s for convenience
    velocity /= 1000  # 1 km/s = 1000 m/s

    return velocity


# Recreate the function to calculate the orbital phase angle for a circular orbit
def calculate_phase_angle(t, t0, p):
    """
    Calculate the orbital phase angle for a circular orbit.

    Parameters:
    t (float): Current time
    t0 (float): Time at which the phase is defined to be zero
    P (float): Orbital period

    Returns:
    float: Phase angle in radians
    """
    import numpy as np

    return 2 * np.pi * (t - t0) / p


def calculate_radial_velocity(v_orb, phase_angle, v_sys):
    """
    Calculate the radial velocity of a planet.

    Parameters:
    v_orb (float): Orbital velocity in km/s
    phase_angle (float): Phase angle in radians
    v_sys (float): Systemic velocity in km/s

    Returns:
    float: Radial velocity in km/s
    """
    import numpy as np  # Importing NumPy for mathematical operations

    # Calculate the radial velocity based on the provided parameters
    vr = v_orb * np.sin(phase_angle) + v_sys

    return vr
