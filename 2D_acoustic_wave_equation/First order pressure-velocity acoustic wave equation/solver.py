import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def acoustic_solver_Dirichlet(
    nz,
    nx,
    wavelet,
    coeff,
    ix_src,
    iz_src,
    dx,
    dz,
    dt,
    rho,
    bulk,
    snapshot_steps,
):
    
    # Basic parameters
    nt = len(wavelet)

    # Half stencil width
    N = len(coeff)

    # Number of snapshots
    ns = len(snapshot_steps)

    inv_rho = 1.0 / rho
    dt_rho = dt * inv_rho
    dt_bulk = dt * bulk

    # Check source position
    if ix_src < N or ix_src >= nx - N:
        raise ValueError("Source x-position is too close to the boundary.")

    if iz_src < N or iz_src >= nz - N:
        raise ValueError("Source z-position is too close to the boundary.")

    # Wavefield allocation

    # Pressure:
    # p(i,j) is located at integer grid points
    p = np.zeros((nz, nx), dtype=np.float64)

    # vx:
    # located at x-half-grid positions
    # p:
    #       p(j-1)     p(j)
    #          o--------o
    #              vx(j)
    vx = np.zeros((nz, nx + 1), dtype=np.float64)

    # vz:
    # located at z-half-grid positions
    vz = np.zeros((nz + 1, nx), dtype=np.float64)

    # Snapshots
    snapshots_p = np.zeros((ns, nz, nx),dtype=np.float64)
    snapshots_vx = np.zeros((ns, nz, nx),dtype=np.float64)
    snapshots_vz = np.zeros((ns, nz, nx),dtype=np.float64)
    snapshot_id = 0

    # Time marching
    for it in range(nt):

        # Progress
        if (it + 1) % 100 == 0 or it == nt - 1:
            print("Time step:",it + 1,"/",nt,"  Time:",(it + 1) * dt,"s")

        # 1. Update vx
        # rho dvx/dt = -dp/dx
        for iz in prange(N, nz - N):
            for ix in range(N, nx - N + 1):

                dpdx = 0.0

                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dpdx += c * (p[iz, ix + m - 1] - p[iz, ix - m])
                dpdx /= dx
                vx[iz, ix] -= dt_rho * dpdx

        # 2. Update vz
        # rho dvz/dt = -dp/dz
        for iz in prange(N, nz - N + 1):
            for ix in range(N, nx - N):

                dpdz = 0.0

                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dpdz += c * (p[iz + m - 1, ix] - p[iz - m, ix])
                dpdz /= dz
                vz[iz, ix] -= dt_rho * dpdz

        # 3. Update pressure
        # dp/dt = -K (dvx/dx + dvz/dz)
        for iz in prange(N, nz - N):
            for ix in range(N, nx - N):

                dvxdx = 0.0
                dvzdz = 0.0

                # dvx / dx
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvxdx += c * (vx[iz, ix + m] - vx[iz, ix - m + 1])
                dvxdx /= dx

                # dvz / dz
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvzdz += c * (vz[iz + m, ix] - vz[iz - m + 1, ix])
                dvzdz /= dz
                # Pressure update
                p[iz, ix] -= dt_bulk * (dvxdx + dvzdz)

        # Add pressure source
        p[iz_src, ix_src] += wavelet[it]

        # Save snapshot
        if snapshot_id < ns:
            if it == snapshot_steps[snapshot_id]:
                for iz in prange(nz):
                    for ix in range(nx):
                        snapshots_p[snapshot_id,iz,ix]  = p[iz, ix]
                        snapshots_vx[snapshot_id,iz,ix] = vx[iz, ix]
                        snapshots_vz[snapshot_id,iz,ix] = vz[iz, ix]
                snapshot_id += 1

    return snapshots_p,snapshots_vx,snapshots_vz



@njit(parallel=True, fastmath=True)
def acoustic_solver_PML(
    nz,
    nx,
    wavelet,
    coeff,
    ix_src,
    iz_src,
    dx,
    dz,
    dt,
    rho,
    bulk,
    snapshot_steps,
    n_pml,
    pml_power=3,
    reflection=1.0e-6,
):

    # =========================================================
    # Basic parameters
    # =========================================================

    nt = len(wavelet)

    # Half stencil width
    N = len(coeff)

    # Number of snapshots
    ns = len(snapshot_steps)

    # Acoustic velocity
    velocity = np.sqrt(bulk / rho)

    inv_rho = 1.0 / rho


    # =========================================================
    # Parameter checking
    # =========================================================

    if n_pml <= N:

        raise ValueError(
            "n_pml must be larger than the FD half stencil width."
        )


    if nx <= 2 * n_pml + 2 * N:

        raise ValueError(
            "nx is too small for the selected PML thickness "
            "and finite-difference order."
        )


    if nz <= 2 * n_pml + 2 * N:

        raise ValueError(
            "nz is too small for the selected PML thickness "
            "and finite-difference order."
        )


    # Source should be located inside the physical domain

    if (
        ix_src < n_pml + N
        or
        ix_src >= nx - n_pml - N
    ):

        raise ValueError(
            "Source x-position is inside or too close to the PML."
        )


    if (
        iz_src < n_pml + N
        or
        iz_src >= nz - n_pml - N
    ):

        raise ValueError(
            "Source z-position is inside or too close to the PML."
        )


    # =========================================================
    # CFL check
    # =========================================================

    coeff_sum = 0.0

    for m in range(N):

        coeff_sum += abs(coeff[m])


    dt_max = 1.0 / (
        velocity
        *
        coeff_sum
        *
        np.sqrt(
            1.0 / (dx * dx)
            +
            1.0 / (dz * dz)
        )
    )


    if dt >= dt_max:

        raise ValueError(
            "The time step may violate the CFL stability condition."
        )


    # =========================================================
    # Wavefield allocation
    # =========================================================

    # Total pressure
    p = np.zeros(
        (nz, nx),
        dtype=np.float64
    )


    # Split pressure fields

    px = np.zeros(
        (nz, nx),
        dtype=np.float64
    )

    pz = np.zeros(
        (nz, nx),
        dtype=np.float64
    )


    # x-direction particle velocity

    vx = np.zeros(
        (nz, nx + 1),
        dtype=np.float64
    )


    # z-direction particle velocity

    vz = np.zeros(
        (nz + 1, nx),
        dtype=np.float64
    )


    # =========================================================
    # Snapshots
    # =========================================================

    snapshots_p = np.zeros(
        (ns, nz, nx),
        dtype=np.float64
    )

    snapshots_vx = np.zeros(
        (ns, nz, nx + 1),
        dtype=np.float64
    )

    snapshots_vz = np.zeros(
        (ns, nz + 1, nx),
        dtype=np.float64
    )


    # =========================================================
    # PML parameters
    # =========================================================

    Lx = n_pml * dx

    Lz = n_pml * dz


    sigma_max_x = (
        -(pml_power + 1)
        *
        velocity
        *
        np.log(reflection)
        /
        (2.0 * Lx)
    )


    sigma_max_z = (
        -(pml_power + 1)
        *
        velocity
        *
        np.log(reflection)
        /
        (2.0 * Lz)
    )


    # =========================================================
    # PML sigma profiles
    #
    # pressure grid
    # =========================================================

    sigma_x_p = np.zeros(
        nx,
        dtype=np.float64
    )

    sigma_z_p = np.zeros(
        nz,
        dtype=np.float64
    )


    # ---------------------------------------------------------
    # sigma_x on pressure grid
    # ---------------------------------------------------------

    for ix in range(nx):

        eta = 0.0


        # Left PML

        if ix < n_pml:

            eta = (
                n_pml - ix
            ) / n_pml


        # Right PML

        elif ix >= nx - n_pml:

            eta = (
                ix
                -
                (nx - n_pml - 1)
            ) / n_pml


        if eta > 1.0:

            eta = 1.0


        if eta > 0.0:

            sigma_x_p[ix] = (
                sigma_max_x
                *
                eta ** pml_power
            )


    # ---------------------------------------------------------
    # sigma_z on pressure grid
    # ---------------------------------------------------------

    for iz in range(nz):

        eta = 0.0


        # Top PML

        if iz < n_pml:

            eta = (
                n_pml - iz
            ) / n_pml


        # Bottom PML

        elif iz >= nz - n_pml:

            eta = (
                iz
                -
                (nz - n_pml - 1)
            ) / n_pml


        if eta > 1.0:

            eta = 1.0


        if eta > 0.0:

            sigma_z_p[iz] = (
                sigma_max_z
                *
                eta ** pml_power
            )


    # =========================================================
    # PML sigma profiles
    #
    # staggered velocity grids
    # =========================================================

    sigma_x_vx = np.zeros(
        nx + 1,
        dtype=np.float64
    )

    sigma_z_vz = np.zeros(
        nz + 1,
        dtype=np.float64
    )


    # ---------------------------------------------------------
    # sigma_x for vx
    #
    # vx is located at x = (ix - 0.5) dx
    # ---------------------------------------------------------

    for ix in range(nx + 1):

        x_pos = (
            ix - 0.5
        ) * dx


        eta = 0.0


        # Left PML

        left_interface = n_pml * dx


        if x_pos < left_interface:

            eta = (
                left_interface - x_pos
            ) / Lx


        # Right PML

        right_interface = (
            nx - n_pml - 1
        ) * dx


        if x_pos > right_interface:

            eta_right = (
                x_pos - right_interface
            ) / Lx


            if eta_right > eta:

                eta = eta_right


        if eta > 1.0:

            eta = 1.0


        if eta > 0.0:

            sigma_x_vx[ix] = (
                sigma_max_x
                *
                eta ** pml_power
            )


    # ---------------------------------------------------------
    # sigma_z for vz
    #
    # vz is located at z = (iz - 0.5) dz
    # ---------------------------------------------------------

    for iz in range(nz + 1):

        z_pos = (
            iz - 0.5
        ) * dz


        eta = 0.0


        # Top PML

        top_interface = n_pml * dz


        if z_pos < top_interface:

            eta = (
                top_interface - z_pos
            ) / Lz


        # Bottom PML

        bottom_interface = (
            nz - n_pml - 1
        ) * dz


        if z_pos > bottom_interface:

            eta_bottom = (
                z_pos - bottom_interface
            ) / Lz


            if eta_bottom > eta:

                eta = eta_bottom


        if eta > 1.0:

            eta = 1.0


        if eta > 0.0:

            sigma_z_vz[iz] = (
                sigma_max_z
                *
                eta ** pml_power
            )


    # =========================================================
    # Precompute PML time coefficients
    # =========================================================

    # Pressure x-direction

    ax_p = np.ones(
        nx,
        dtype=np.float64
    )

    bx_p = np.ones(
        nx,
        dtype=np.float64
    ) * dt


    for ix in range(nx):

        sigma = sigma_x_p[ix]


        if sigma > 0.0:

            ax_p[ix] = np.exp(
                -sigma * dt
            )

            bx_p[ix] = (
                1.0 - ax_p[ix]
            ) / sigma


    # Pressure z-direction

    az_p = np.ones(
        nz,
        dtype=np.float64
    )

    bz_p = np.ones(
        nz,
        dtype=np.float64
    ) * dt


    for iz in range(nz):

        sigma = sigma_z_p[iz]


        if sigma > 0.0:

            az_p[iz] = np.exp(
                -sigma * dt
            )

            bz_p[iz] = (
                1.0 - az_p[iz]
            ) / sigma


    # vx

    ax_vx = np.ones(
        nx + 1,
        dtype=np.float64
    )

    bx_vx = np.ones(
        nx + 1,
        dtype=np.float64
    ) * dt


    for ix in range(nx + 1):

        sigma = sigma_x_vx[ix]


        if sigma > 0.0:

            ax_vx[ix] = np.exp(
                -sigma * dt
            )

            bx_vx[ix] = (
                1.0 - ax_vx[ix]
            ) / sigma


    # vz

    az_vz = np.ones(
        nz + 1,
        dtype=np.float64
    )

    bz_vz = np.ones(
        nz + 1,
        dtype=np.float64
    ) * dt


    for iz in range(nz + 1):

        sigma = sigma_z_vz[iz]


        if sigma > 0.0:

            az_vz[iz] = np.exp(
                -sigma * dt
            )

            bz_vz[iz] = (
                1.0 - az_vz[iz]
            ) / sigma


    # =========================================================
    # Time marching
    # =========================================================

    snapshot_id = 0


    for it in range(nt):


        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if (it + 1) % 100 == 0 or it == nt - 1:

            print(
                "Time step:",
                it + 1,
                "/",
                nt,
                " Time:",
                (it + 1) * dt,
                "s"
            )


        # =====================================================
        # 1. Update vx
        #
        # dvx/dt + sigma_x vx
        # = -(1/rho) dp/dx
        # =====================================================

        for iz in prange(
            N,
            nz - N
        ):

            for ix in range(
                N,
                nx - N + 1
            ):

                dpdx = 0.0


                for m in range(
                    1,
                    N + 1
                ):

                    c = coeff[m - 1]


                    dpdx += c * (
                        p[
                            iz,
                            ix + m - 1
                        ]
                        -
                        p[
                            iz,
                            ix - m
                        ]
                    )


                dpdx /= dx


                vx[iz, ix] = (
                    ax_vx[ix]
                    *
                    vx[iz, ix]
                    -
                    bx_vx[ix]
                    *
                    inv_rho
                    *
                    dpdx
                )


        # =====================================================
        # 2. Update vz
        #
        # dvz/dt + sigma_z vz
        # = -(1/rho) dp/dz
        # =====================================================

        for iz in prange(
            N,
            nz - N + 1
        ):

            for ix in range(
                N,
                nx - N
            ):

                dpdz = 0.0


                for m in range(
                    1,
                    N + 1
                ):

                    c = coeff[m - 1]


                    dpdz += c * (
                        p[
                            iz + m - 1,
                            ix
                        ]
                        -
                        p[
                            iz - m,
                            ix
                        ]
                    )


                dpdz /= dz


                vz[iz, ix] = (
                    az_vz[iz]
                    *
                    vz[iz, ix]
                    -
                    bz_vz[iz]
                    *
                    inv_rho
                    *
                    dpdz
                )


        # =====================================================
        # 3. Update px
        #
        # dpx/dt + sigma_x px
        # = -K dvx/dx
        # =====================================================

        for iz in prange(
            N,
            nz - N
        ):

            for ix in range(
                N,
                nx - N
            ):

                dvxdx = 0.0


                for m in range(
                    1,
                    N + 1
                ):

                    c = coeff[m - 1]


                    dvxdx += c * (
                        vx[
                            iz,
                            ix + m
                        ]
                        -
                        vx[
                            iz,
                            ix - m + 1
                        ]
                    )


                dvxdx /= dx


                px[iz, ix] = (
                    ax_p[ix]
                    *
                    px[iz, ix]
                    -
                    bx_p[ix]
                    *
                    bulk
                    *
                    dvxdx
                )


        # =====================================================
        # 4. Update pz
        #
        # dpz/dt + sigma_z pz
        # = -K dvz/dz
        # =====================================================

        for iz in prange(
            N,
            nz - N
        ):

            for ix in range(
                N,
                nx - N
            ):

                dvzdz = 0.0


                for m in range(
                    1,
                    N + 1
                ):

                    c = coeff[m - 1]


                    dvzdz += c * (
                        vz[
                            iz + m,
                            ix
                        ]
                        -
                        vz[
                            iz - m + 1,
                            ix
                        ]
                    )


                dvzdz /= dz


                pz[iz, ix] = (
                    az_p[iz]
                    *
                    pz[iz, ix]
                    -
                    bz_p[iz]
                    *
                    bulk
                    *
                    dvzdz
                )


        # =====================================================
        # 5. Add source
        #
        # p = px + pz
        # =====================================================

        source = 0.5 * wavelet[it]


        px[
            iz_src,
            ix_src
        ] += source


        pz[
            iz_src,
            ix_src
        ] += source


        # =====================================================
        # 6. Calculate total pressure
        # =====================================================

        for iz in prange(
            N,
            nz - N
        ):

            for ix in range(
                N,
                nx - N
            ):

                p[iz, ix] = (
                    px[iz, ix]
                    +
                    pz[iz, ix]
                )


        # =====================================================
        # 7. Save snapshots
        # =====================================================

        if snapshot_id < ns:

            if (
                it
                ==
                snapshot_steps[
                    snapshot_id
                ]
            ):

                for iz in prange(nz):

                    for ix in range(nx):

                        snapshots_p[
                            snapshot_id,
                            iz,
                            ix
                        ] = p[iz, ix]


                for iz in prange(nz):

                    for ix in range(nx + 1):

                        snapshots_vx[
                            snapshot_id,
                            iz,
                            ix
                        ] = vx[iz, ix]


                for iz in prange(nz + 1):

                    for ix in range(nx):

                        snapshots_vz[
                            snapshot_id,
                            iz,
                            ix
                        ] = vz[iz, ix]


                snapshot_id += 1


    return (
        snapshots_p,
        snapshots_vx,
        snapshots_vz
    )