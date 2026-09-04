from scipy.ndimage import gaussian_filter

# Gaussian smoothing
def smoothing_velocity_model(velocity, sigma_x=5.0, sigma_z=5.0):
    velocity_smooth = gaussian_filter(
        velocity,
        sigma=(sigma_z, sigma_x),
        mode="nearest"
    )
    return velocity_smooth

