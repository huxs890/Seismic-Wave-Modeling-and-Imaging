import numpy as np
from numba import njit, prange
# ============================================================
# First-order acoustic wave equation
#   rho * dvx/dt = - dp/dx
#   rho * dvz/dt = - dp/dz
#          dp/dt = - K (dvx/dx + dvz/dz)
# Staggered-grid arrangement:
#             vz
#              |
#              |
#      vx ---- p ---- vx
#              |
#              |
#             vz
# p    : (nz, nx)
# vx   : (nz, nx+1)
# vz   : (nz+1, nx)
#
# Dirichlet boundary:
# p = 0 near computational boundaries
# ============================================================
@njit(parallel=True, fastmath=True)
def _acoustic_solver_dirichlet(
    rho,
    bulk,
    dx,
    dz,
    dt,
    coeff,
    ix_src,
    iz_src,
    wavelet,
    ix_receivers,
    iz_receivers,
    snapshot_steps,
):
    
    # ============================================================
    # 1. Basic parameters
    # ============================================================
    nz, nx = rho.shape
    nt = len(wavelet)
    N = len(coeff)           # Half stencil width
    n_snapshots = len(snapshot_steps) # Number of snapshots
    n_receivers = len(ix_receivers)
    # ============================================================
    # 2. Check model
    # ============================================================ 
    if bulk.shape[0] != nz or bulk.shape[1] != nx:
        raise ValueError("rho and bulk must have the same shape.")
    if nx <= 2 * N or nz <= 2 * N:
        raise ValueError("Model is too small for the selected FD order.")
    # ============================================================
    # 3. Check source position
    # ============================================================ 
    if ix_src < N or ix_src >= nx - N:
        raise ValueError("Source x-position is too close to the boundary.")
    if iz_src < N or iz_src >= nz - N:
        raise ValueError("Source z-position is too close to the boundary.")
    # ============================================================
    # 4. Check Receivers
    # ============================================================
    if len(ix_receivers) != len(iz_receivers):
        raise ValueError("ix_receivers and iz_receivers must have the same length.")
    for ir in range(n_receivers):
        if ix_receivers[ir] < N or ix_receivers[ir] >= nx - N:
            raise ValueError("Receiver x-position is too close to the boundary.")
        if iz_receivers[ir] < N or iz_receivers[ir] >= nz - N:
            raise ValueError("Receiver z-position is too close to the boundary.")
    # ============================================================
    # 5. Buoyancy interpolation
    # ============================================================
    inv_rho_vx = np.zeros((nz, nx + 1),dtype=np.float64)
    inv_rho_vz = np.zeros((nz + 1, nx),dtype=np.float64)
    # ------------------------------------------------------------
    # 1/rho at vx locations
    # ------------------------------------------------------------
    for iz in prange(nz):
        inv_rho_vx[iz, 0] = 1.0 / rho[iz, 0]
        for ix in range(1, nx):
            inv_rho_vx[iz, ix] = 0.5 * (1.0 / rho[iz, ix - 1] + 1.0 / rho[iz, ix])
            inv_rho_vx[iz, nx] = (1.0 / rho[iz, nx - 1])
    # ------------------------------------------------------------
    # 1/rho at vz locations
    # ------------------------------------------------------------
    for ix in prange(nx):
        inv_rho_vz[0, ix] = 1.0 / rho[0, ix]
        for iz in range(1, nz):
            inv_rho_vz[iz, ix] = 0.5 * (1.0 / rho[iz - 1, ix] + 1.0 / rho[iz, ix])
            inv_rho_vz[nz, ix] = 1.0 / rho[nz - 1, ix]
    # ============================================================
    # 6. Wavefield allocation
    # ============================================================
    # ------------------------------
    # Pressure
    # p(i,j)
    # integer grid
    # ------------------------------
    p = np.zeros((nz, nx),dtype=np.float64)
    # --------------------------------------------------------
    # Horizontal particle velocity
    # p(j-1) ---- vx(j) ---- p(j)
    # therefore vx has nx+1 grid points
    # --------------------------------------------------------
    vx = np.zeros((nz, nx + 1),dtype=np.float64)
    # --------------------------------------------------------
    # Vertical particle velocity
    #      p(i-1)
    #        |
    #       vz(i)
    #        |
    #       p(i)
    # therefore vz has nz+1 grid points
    # --------------------------------------------------------
    vz = np.zeros((nz + 1, nx),dtype=np.float64)
    # ============================================================
    # 7. Seismogram
    # dimension: (time, receiver)
    # ============================================================
    seismogram_p = np.zeros((nt, n_receivers),dtype=np.float64)
    # ============================================================
    # 8. snapshots
    # ============================================================
    snapshots_p = np.zeros((n_snapshots, nz, nx),dtype=np.float64)
    snapshot_id = 0
    # ============================================================
    # 9. Time loop
    # ============================================================
    for it in range(nt-1):
        # Progress
        if (it + 1) % 100 == 0:
            print("Time step:",it + 1,"/",nt,"  Time:",(it + 1) * dt,"s")
        # -------------------------------------------------
        # 1). Update vx
        # rho dvx/dt = -dp/dx
        #        p(i,j-1)      p(i,j)
        #            o----------o
        #                 |
        #               vx(i,j)
        # -------------------------------------------------
        for iz in prange(N, nz - N):
            for ix in range(N, nx - N + 1):
                dpdx = 0.0
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dpdx += c * (p[iz, ix + m - 1] - p[iz, ix - m])
                dpdx /= dx
                vx[iz, ix] -= dt * inv_rho_vx[iz, ix] * dpdx
        # -------------------------------------------------
        # 2). Update vz
        # rho dvz/dt = -dp/dz
        # -------------------------------------------------
        for iz in prange(N, nz - N + 1):
            for ix in range(N, nx - N):
                dpdz = 0.0
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dpdz += c * (p[iz + m - 1, ix] - p[iz - m, ix])
                dpdz /= dz
                vz[iz, ix] -= dt * inv_rho_vz[iz, ix] * dpdz
        # -------------------------------------------------
        # 3). Update pressure
        # dp/dt = -K (dvx/dx + dvz/dz)
        # -------------------------------------------------
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
                p[iz, ix] -= dt * bulk[iz,ix] * (dvxdx + dvzdz)
        # -------------------------------------------------
        # 4). Add pressure source
        # -------------------------------------------------
        p[iz_src, ix_src] += dt * wavelet[it]
        # -------------------------------------------------
        # 5). Record receivers
        # -------------------------------------------------
        for ir in range(n_receivers):
            ix_rec = ix_receivers[ir]
            iz_rec = iz_receivers[ir]
            seismogram_p[it+1, ir] = p[iz_rec,ix_rec]
        # -------------------------------------------------
        # 6). Save pressure snapshot
        # -------------------------------------------------    
        if snapshot_id < n_snapshots:
            if (it+1) == snapshot_steps[snapshot_id]:
                for iz in prange(nz):
                    for ix in range(nx):
                        snapshots_p[snapshot_id,iz,ix]  = p[iz, ix]
                        # snapshots_vx[snapshot_id,iz,ix] = vx[iz, ix]
                        # snapshots_vz[snapshot_id,iz,ix] = vz[iz, ix]
                snapshot_id += 1

    return seismogram_p,snapshots_p