import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def elastic_solver_Dirichlet(
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