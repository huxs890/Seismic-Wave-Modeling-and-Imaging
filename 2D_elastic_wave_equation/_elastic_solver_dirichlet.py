import numpy as np
from numba import njit, prange
# ============================================================
# 2-D VTI elastic wave equation
# First-order stress-velocity formulation
#
# rho * dvx/dt = dsxx/dx + dsxz/dz
# rho * dvz/dt = dsxz/dx + dszz/dz
#
# dsxx/dt = C11 * dvx/dx + C13 * dvz/dz
# dszz/dt = C13 * dvx/dx + C33 * dvz/dz
#
# dsxz/dt = C55 * (dvx/dz + dvz/dx)
#
# Staggered-grid arrangement
#
# sxx, szz : (nz, nx)
#
# vx       : (nz, nx+1)
#
# vz       : (nz+1, nx)
#
# sxz      : (nz+1, nx+1)
#
# Spatial arrangement:
#
#               sxz -------- vz -------- sxz
#                |                        |
#                |                        |
#                vx      sxx,szz          vx
#                |                        |
#                |                        |
#               sxz -------- vz -------- sxz
#
# Dirichlet truncation:
# outer N grid points are not updated and remain zero.
# ============================================================
@njit(parallel=True, fastmath=True)
def _elastic_vti_solver_dirichlet(
    rho,
    c11,
    c13,
    c33,
    c55,
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
    # ========================================================
    # 1. Basic parameters
    # ========================================================
    nz, nx = rho.shape
    nt = len(wavelet)
    N = len(coeff) # Half stencil width
    n_receivers = len(ix_receivers)
    n_snapshots = len(snapshot_steps)
    # ========================================================
    # 2. Density interpolation
    # buoyancy = 1/rho
    # rho is defined at sxx/szz positions.
    # vx and vz are located at half-grid positions.
    # ========================================================
    inv_rho_vx = np.zeros((nz, nx + 1),dtype=np.float32)
    inv_rho_vz = np.zeros((nz + 1, nx),dtype=np.float32)
    # --------------------------------------------------------
    # 1/rho -> vx
    # --------------------------------------------------------
    for iz in prange(nz):
        inv_rho_vx[iz, 0] = 1.0 / rho[iz, 0]
        for ix in range(1, nx):
            inv_rho_vx[iz, ix] = 0.5 * (1.0 / rho[iz, ix - 1] + 1.0 / rho[iz, ix])
        inv_rho_vx[iz, nx] = 1.0 / rho[iz, nx - 1]
    # --------------------------------------------------------
    # 1/rho -> vz
    # --------------------------------------------------------
    for ix in prange(nx):
        inv_rho_vz[0, ix] = 1.0 / rho[0, ix]
        for iz in range(1, nz):
            inv_rho_vz[iz, ix] = 0.5 * (1.0 / rho[iz - 1, ix]+1.0 / rho[iz, ix])
        inv_rho_vz[nz, ix] = 1.0 / rho[nz - 1, ix]
    # ========================================================
    # 3. Interpolate C55 to sxz grid
    # C55 is originally defined at cell centers.
    # sxz is located at x-z half-grid position.
    # ========================================================
    c55_xz = np.zeros((nz + 1, nx + 1),dtype=np.float32)
    for iz in prange(1, nz):
        for ix in range(1, nx):
            c55_xz[iz, ix] = 0.25 * (c55[iz - 1, ix - 1]+c55[iz - 1, ix]+c55[iz, ix - 1]+c55[iz, ix])
    # ========================================================
    # 4. Wavefield allocation
    # ========================================================
    # --------------------------------------------------------
    # Normal stresses
    # --------------------------------------------------------
    sxx = np.zeros((nz, nx),dtype=np.float32)
    szz = np.zeros((nz, nx),dtype=np.float32)
    # --------------------------------------------------------
    # Shear stress
    # --------------------------------------------------------
    sxz = np.zeros((nz + 1, nx + 1),dtype=np.float32)
    # --------------------------------------------------------
    # Particle velocities
    # --------------------------------------------------------
    vx = np.zeros((nz, nx + 1),dtype=np.float32)
    vz = np.zeros((nz + 1, nx),dtype=np.float32)
    vx_old = np.zeros((nz, nx + 1),dtype=np.float32)
    vz_old = np.zeros((nz + 1, nx),dtype=np.float32)
    # ========================================================
    # 5. Seismograms
    # velocities are interpolated back to cell centers.
    # ========================================================
    seismogram_vx = np.zeros((nt, n_receivers),dtype=np.float32)
    seismogram_vz = np.zeros((nt, n_receivers),dtype=np.float32)
    # ========================================================
    # 6. Snapshots
    # ========================================================
    snapshots_vx = np.zeros((n_snapshots, nz, nx),dtype=np.float32)
    snapshots_vz = np.zeros((n_snapshots, nz, nx),dtype=np.float32)
    snapshot_id = 0
    # ========================================================
    # 7. Time loop
    # stress^n
    #       ->
    # velocity^(n+1/2)
    #       ->
    # stress^(n+1)
    # ========================================================
    for it in range(nt - 1):
        # ====================================================
        # 1). Update vx
        # rho dvx/dt = dsxx/dx + dsxz/dz
        # ====================================================
        for iz in prange(N, nz - N):
            for ix in range(N, nx - N + 1):
                dsxx_dx = 0.0
                dsxz_dz = 0.0
                # --------------------------------------------
                # dsxx / dx
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dsxx_dx += c * (sxx[iz, ix + m - 1]-sxx[iz, ix - m])
                dsxx_dx /= dx
                # --------------------------------------------
                # dsxz / dz
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dsxz_dz += c * (sxz[iz + m, ix]-sxz[iz - m + 1, ix])
                dsxz_dz /= dz
                # --------------------------------------------
                # vx update
                # --------------------------------------------
                vx[iz, ix] += dt * inv_rho_vx[iz, ix]*(dsxx_dx + dsxz_dz)
        # ====================================================
        # 2). Update vz
        # rho dvz/dt = dsxz/dx + dszz/dz
        # ====================================================
        for iz in prange(N, nz - N + 1):
            for ix in range(N, nx - N):
                dsxz_dx = 0.0
                dszz_dz = 0.0
                # --------------------------------------------
                # dsxz / dx
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dsxz_dx += c * (sxz[iz, ix + m]-sxz[iz, ix - m + 1])
                dsxz_dx /= dx
                # --------------------------------------------
                # dszz / dz
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dszz_dz += c * (szz[iz + m - 1, ix]-szz[iz - m, ix])
                dszz_dz /= dz
                # --------------------------------------------
                # vz update
                # --------------------------------------------
                vz[iz, ix] += dt * inv_rho_vz[iz, ix] * (dsxz_dx + dszz_dz)
        # ====================================================
        # 3). Update sxx and szz
        # dsxx/dt = C11 dvx/dx + C13 dvz/dz
        # dszz/dt = C13 dvx/dx + C33 dvz/dz
        # ====================================================
        for iz in prange(N, nz - N):
            for ix in range(N, nx - N):
                dvx_dx = 0.0
                dvz_dz = 0.0
                # --------------------------------------------
                # dvx / dx
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvx_dx += c * (vx[iz, ix + m]-vx[iz, ix - m + 1])
                dvx_dx /= dx
                # --------------------------------------------
                # dvz / dz
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvz_dz += c * (vz[iz + m, ix]-vz[iz - m + 1, ix])
                dvz_dz /= dz
                # --------------------------------------------
                # sxx
                # --------------------------------------------
                sxx[iz, ix] += dt * (c11[iz, ix] * dvx_dx+c13[iz, ix] * dvz_dz)
                # --------------------------------------------
                # szz
                # --------------------------------------------
                szz[iz, ix] += dt * (c13[iz, ix] * dvx_dx+c33[iz, ix] * dvz_dz)
        # ====================================================
        # 4). Update sxz
        # dsxz/dt =
        # C55 * (dvx/dz + dvz/dx)
        # ====================================================
        for iz in prange(N, nz - N + 1):
            for ix in range(N, nx - N + 1):
                dvx_dz = 0.0
                dvz_dx = 0.0
                # --------------------------------------------
                # dvx / dz
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvx_dz += c * (vx[iz + m - 1, ix]-vx[iz - m, ix])
                dvx_dz /= dz
                # --------------------------------------------
                # dvz / dx
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvz_dx += c * (vz[iz, ix + m - 1]-vz[iz, ix - m])
                dvz_dx /= dx
                # --------------------------------------------
                # sxz
                # --------------------------------------------
                sxz[iz, ix] += dt*c55_xz[iz, ix]*(dvx_dz + dvz_dx)
        # ====================================================
        # 5). Isotropic explosive source
        # Add equally to sxx and szz.
        # This mainly excites the qP wave.
        # ====================================================
        source = dt * wavelet[it]
        sxx[iz_src,ix_src] += source
        szz[iz_src,ix_src] += source
        # ====================================================
        # Current time level
        # ====================================================
        step = it
        # ====================================================
        # 6). Record receivers
        # interpolate vx / vz to cell centers
        # ====================================================
        for ir in range(n_receivers):
            ix_rec = ix_receivers[ir]
            iz_rec = iz_receivers[ir]
            vx_time = 0.5 * (vx_old[iz_rec, ix_rec]+vx[iz_rec, ix_rec])
            vx_time_right = 0.5 * (vx_old[iz_rec, ix_rec + 1]+vx[iz_rec, ix_rec + 1])
            vz_time = 0.5* (vz_old[iz_rec, ix_rec]+vz[iz_rec, ix_rec])
            vz_time_down = 0.5 * (vz_old[iz_rec + 1, ix_rec]+vz[iz_rec + 1, ix_rec])
            seismogram_vx[step, ir] = 0.5*(vx_time + vx_time_right)
            seismogram_vz[step, ir] = 0.5*(vz_time + vz_time_down)
        # ====================================================
        # 7). Save snapshots
        # ====================================================
        if snapshot_id < n_snapshots:
            if step == snapshot_steps[snapshot_id]:
                for iz in prange(nz):
                    for ix in range(nx):
                        vx_left = 0.5 * (vx_old[iz, ix] + vx[iz, ix])
                        vx_right = 0.5 * (vx_old[iz, ix + 1] + vx[iz, ix + 1])
                        snapshots_vx[snapshot_id,iz,ix] = 0.5 * (vx_left+vx_right)

                        vz_up = 0.5 * (vz_old[iz, ix]+vz[iz, ix])
                        vz_down = 0.5 * (vz_old[iz + 1, ix]+vz[iz + 1, ix])
                        snapshots_vz[snapshot_id,iz,ix] = np.float32(0.5) * (vz_up+vz_down)
                snapshot_id += 1
        # ====================================================
        # Progress
        # ====================================================
        if step % 100 == 0 or step == nt - 1:
            print("Time step:",step,"/",nt - 1," Time:",step * dt,"s")

    # ========================================================
    # Output
    # ========================================================
    return seismogram_vx,seismogram_vz,snapshots_vx,snapshots_vz