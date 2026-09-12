import numpy as np

# ============================================================
    # Geometry of extended computational domain
    #        <----------- nx_ext ----------->
    #        +-----------------------------+
    #        |          Top PML            |
    #        |        sigma_z > 0          |
    #        |                             |
    #        |   +---------------------+   |
    #        |   |                     |   |
    #        |   |   Physical domain   |   |
    # Left   |   |      nz x nx        |   | Right
    # PML    |   |                     |   | PML
    # sx>0   |   +---------------------+   | sx>0
    #        |                             |
    #        |        Bottom PML           |
    #        |        sigma_z > 0          |
    #        +-----------------------------+
    # Physical model: rho.shape = (nz, nx)
    # Extended model: rho_ext.shape = (nz + 2*n_abs,nx + 2*n_abs)
    # Physical region inside extended model = rho_ext[n_abs : n_abs + nz,n_abs : n_abs + nx]
    # Therefore:
    # iz_ext = iz + n_abs
    # ix_ext = ix + n_abs
# =================================================================

def PML_boundary(
    rho,
    bulk,
    dx,
    dz,
    n_abs=40,
    reflection=1e-6,
    power=2,
):
    # ============================================================
    # 1. Extend physical model
    # ============================================================
    rho_ext = np.pad(rho,((n_abs, n_abs), (n_abs, n_abs)),mode="edge")
    bulk_ext = np.pad(bulk,((n_abs, n_abs), (n_abs, n_abs)),mode="edge")
    nz_ext, nx_ext = rho_ext.shape
    # ============================================================
    # 2. Maximum acoustic velocity
    # vp = sqrt(K / rho)
    # ============================================================
    velocity_ext = np.sqrt(bulk_ext / rho_ext)
    vmax = np.max(velocity_ext)
    # ============================================================
    # 3. PML thickness
    # ============================================================
    Lx = n_abs * dx
    Lz = n_abs * dz
    # ============================================================
    # 4. Maximum damping
    # ============================================================
    sigma_max_x = -(power + 1)* vmax* np.log(reflection) / (2.0 * Lx)
    sigma_max_z = -(power + 1)* vmax* np.log(reflection) / (2.0 * Lz)
    # ============================================================
    # 5. 1D damping profiles
    # ============================================================
    sigma_x_1d = np.zeros(nx_ext,dtype=np.float64)
    sigma_z_1d = np.zeros(nz_ext,dtype=np.float64)
    for i in range(n_abs):
        # eta = 1 at outer boundary
        # eta -> 0 near physical domain
        eta = (n_abs - i) / n_abs
        sigma_x_value = sigma_max_x* eta**power
        sigma_z_value = sigma_max_z * eta**power
        # Left / right PML
        sigma_x_1d[i] = sigma_x_value
        sigma_x_1d[nx_ext - 1 - i] = sigma_x_value
        # Top / bottom PML
        sigma_z_1d[i] = sigma_z_value
        sigma_z_1d[nz_ext - 1 - i] = sigma_z_value
    # ============================================================
    # 6. Convert to 2D damping fields
    # ============================================================
    sigma_x = np.tile(sigma_x_1d,(nz_ext, 1))
    sigma_z = np.tile(sigma_z_1d[:, np.newaxis],(1, nx_ext))

    return rho_ext, bulk_ext, sigma_x, sigma_z