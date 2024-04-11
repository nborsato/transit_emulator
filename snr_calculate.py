def calculate_snr_based_on_v_mag(m, snr_ref=72, m_ref=10):
    import math
    """
    Calculate the new SNR for a star with a different magnitude.

    Parameters:
        snr_ref (float): The reference SNR
        m_ref (float): The reference magnitude
        m (float): The new magnitude for which to calculate the SNR

    Returns:
        float: The new SNR for the star with magnitude m_new
    """
    return snr_ref * math.sqrt(10 ** (-0.4 * (m - m_ref)))
