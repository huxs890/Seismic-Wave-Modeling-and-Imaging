import numpy as np
# ============================================================
# Geometry of extended computational domain
#
#        <------------- nx_ext ------------->
#
#        +----------------------------------+
#        |             Top PML              |
#        |          sigma_z > 0             |
#        |                                  |
#        |    +------------------------+    |
#        |    |                        |    |
#        |    |    Physical domain     |    |
# Left   |    |       nz x nx          |    | Right
# PML    |    |                        |    | PML
# sx>0   |    +------------------------+    | sx>0
#        |                                  |
#        |            Bottom PML            |
#        |          sigma_z > 0             |
#        +----------------------------------+
#
# Physical model:
#
# rho.shape = (nz, nx)
#
# Extended model:
#
# rho_ext.shape =
# (nz + 2*n_abs, nx + 2*n_abs)
#
# Physical region:
#
# [n_abs:n_abs+nz, n_abs:n_abs+nx]
#
# Coordinate transformation:
#
# iz_ext = iz + n_abs
# ix_ext = ix + n_abs
#
# ============================================================
def VTI_PML_boundary(
    rho,
    c11,
    c13,
    c33,
    c55,
    dx,
    dz,
    n_abs=40,
    reflection=1e-6,
    power=2,
):
    # ========================================================
    # 1. Data type
    # ========================================================

    rho = np.asarray(
        rho,
        dtype=np.float32
    )

    c11 = np.asarray(
        c11,
        dtype=np.float32
    )

    c13 = np.asarray(
        c13,
        dtype=np.float32
    )

    c33 = np.asarray(
        c33,
        dtype=np.float32
    )

    c55 = np.asarray(
        c55,
        dtype=np.float32
    )

    dx = np.float32(dx)
    dz = np.float32(dz)

    n_abs = np.int32(n_abs)

    # ========================================================
    # 2. Extend physical model
    #
    # Edge padding avoids introducing artificial material
    # interfaces inside the PML.
    # ========================================================

    pad = (
        (n_abs, n_abs),
        (n_abs, n_abs)
    )

    rho_ext = np.pad(
        rho,
        pad,
        mode="edge"
    ).astype(np.float32)

    c11_ext = np.pad(
        c11,
        pad,
        mode="edge"
    ).astype(np.float32)

    c13_ext = np.pad(
        c13,
        pad,
        mode="edge"
    ).astype(np.float32)

    c33_ext = np.pad(
        c33,
        pad,
        mode="edge"
    ).astype(np.float32)

    c55_ext = np.pad(
        c55,
        pad,
        mode="edge"
    ).astype(np.float32)

    nz_ext, nx_ext = rho_ext.shape

    # ========================================================
    # 3. Estimate maximum VTI wave velocity
    #
    # Horizontal qP velocity:
    #
    # vp_x = sqrt(C11 / rho)
    #
    # Vertical qP velocity:
    #
    # vp_z = sqrt(C33 / rho)
    #
    # For ordinary weak/moderate VTI media, the larger of
    # these provides a practical characteristic velocity
    # for constructing the PML damping profile.
    # ========================================================

    vp_x = np.sqrt(
        c11_ext / rho_ext
    )

    vp_z = np.sqrt(
        c33_ext / rho_ext
    )

    vmax = np.float32(
        max(
            np.max(vp_x),
            np.max(vp_z)
        )
    )

    # ========================================================
    # 4. Physical PML thickness
    # ========================================================

    Lx = np.float32(n_abs) * dx
    Lz = np.float32(n_abs) * dz

    # ========================================================
    # 5. Maximum damping coefficients
    #
    # Polynomial damping:
    #
    # sigma(x) = sigma_max * eta^power
    #
    # sigma_max is estimated from the desired reflection
    # coefficient.
    # ========================================================

    reflection = np.float32(reflection)

    sigma_max_x = np.float32(
        -(power + 1)
        * vmax
        * np.log(reflection)
        /
        (2.0 * Lx)
    )

    sigma_max_z = np.float32(
        -(power + 1)
        * vmax
        * np.log(reflection)
        /
        (2.0 * Lz)
    )

    # ========================================================
    # 6. One-dimensional damping profiles
    #
    # sigma_x:
    #
    # nonzero only in left/right PML
    #
    # sigma_z:
    #
    # nonzero only in top/bottom PML
    # ========================================================

    sigma_x_1d = np.zeros(
        nx_ext,
        dtype=np.float32
    )

    sigma_z_1d = np.zeros(
        nz_ext,
        dtype=np.float32
    )

    for i in range(n_abs):

        # eta = 1 at outer boundary
        # eta -> 0 toward physical domain

        eta = np.float32(
            (n_abs - i)
            /
            n_abs
        )

        sigma_x_value = (
            sigma_max_x
            *
            eta**power
        )

        sigma_z_value = (
            sigma_max_z
            *
            eta**power
        )

        # Left / right PML

        sigma_x_1d[i] = sigma_x_value

        sigma_x_1d[
            nx_ext - 1 - i
        ] = sigma_x_value

        # Top / bottom PML

        sigma_z_1d[i] = sigma_z_value

        sigma_z_1d[
            nz_ext - 1 - i
        ] = sigma_z_value

    # ========================================================
    # 7. Convert 1-D profiles to 2-D damping fields
    #
    # sigma_x = sigma_x(x)
    #
    # sigma_z = sigma_z(z)
    #
    # At corner PML regions:
    #
    # sigma_x > 0
    # sigma_z > 0
    #
    # therefore waves are damped simultaneously in both
    # coordinate directions.
    # ========================================================

    sigma_x = np.tile(
        sigma_x_1d,
        (nz_ext, 1)
    ).astype(np.float32)

    sigma_z = np.tile(
        sigma_z_1d[:, np.newaxis],
        (1, nx_ext)
    ).astype(np.float32)

    # ========================================================
    # Output
    # ========================================================

    return (
        rho_ext,
        c11_ext,
        c13_ext,
        c33_ext,
        c55_ext,
        sigma_x,
        sigma_z,
    )