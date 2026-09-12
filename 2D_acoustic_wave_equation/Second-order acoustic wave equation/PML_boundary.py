import numpy as np

def PML_boundary(
    velocity,
    dx,
    dz,
    n_abs=40,
    reflection=1e-6,
    power=2,
):

    nz, nx = velocity.shape

    # Extend velocity model
    velocity_ext = np.pad(
        velocity,
        pad_width=((n_abs, n_abs),(n_abs, n_abs)),
        mode="edge"
    )
    nz_ext, nx_ext = velocity_ext.shape

    # Maximum velocity
    vmax = np.max(velocity_ext)

    # Physical thickness of absorbing layer
    Lx = n_abs * dx
    Lz = n_abs * dz

    # Maximum damping coefficients
    sigma_max_x = -(power + 1) * vmax * np.log(reflection) / (2.0 * Lx)
    sigma_max_z = -(power + 1) * vmax * np.log(reflection) / (2.0 * Lz)
    
    # 1D damping profiles
    sigma_x = np.zeros(nx_ext)
    sigma_z = np.zeros(nz_ext)
    for i in range(n_abs):
        # normalized distance:
        # 0 near physical domain
        # 1 at outer boundary
        eta = (n_abs - i) / n_abs
        value_x = sigma_max_x * eta**power
        value_z = sigma_max_z * eta**power
        # left / right
        sigma_x[i] = value_x
        sigma_x[nx_ext - 1 - i] = value_x
        # top / bottom
        sigma_z[i] = value_z
        sigma_z[nz_ext - 1 - i] = value_z

    # 2D damping field
    sigma = sigma_x[np.newaxis, :] + sigma_z[:, np.newaxis]

    return velocity_ext, sigma