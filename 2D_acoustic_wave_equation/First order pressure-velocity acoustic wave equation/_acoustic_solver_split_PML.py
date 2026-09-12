import numpy as np
from numba import njit, prange
# ============================================================
# First-order acoustic wave equation with split-field PML
# dvx/dt + sigma_x * vx = -(1/rho) * dp/dx
# dvz/dt + sigma_z * vz = -(1/rho) * dp/dz
# dpx/dt + sigma_x * px = -K * dvx/dx
# dpz/dt + sigma_z * pz = -K * dvz/dz
# p = px + pz
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
# PML boundary:
# sigma_x, sigma_z are defined at pressure grid points.
# ============================================================
@njit(parallel=True, fastmath=True)
def _acoustic_solver_split_PML(
    rho_ext,
    bulk_ext,
    sigma_x,
    sigma_z,
    n_abs,
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
    nz_ext, nx_ext = rho_ext.shape
    nz = nz_ext - 2 * n_abs
    nx = nx_ext - 2 * n_abs
    nt = len(wavelet)
    N = len(coeff)           # Half stencil width
    n_snapshots = len(snapshot_steps) # Number of snapshots
    n_receivers = len(ix_receivers)
    # ============================================================
    # 2. Check model 
    # ============================================================ 
    if bulk_ext.shape != rho_ext.shape:
        raise ValueError("bulk_extent and rho_extent must have the same shape.")
    if sigma_x.shape != rho_ext.shape:
        raise ValueError("sigma_x must have the same shape as rho_extent.")
    if sigma_z.shape != rho_ext.shape:
        raise ValueError("sigma_z must have the same shape as rho_extent.")
    if nz <= 0 or nx <= 0:
        raise ValueError("n_abs is too large for the extended model.")
    if len(ix_receivers) != len(iz_receivers):
        raise ValueError("ix_receivers and iz_receivers must have the same length.")
    # ============================================================
    # 3. Physical coordinates -> extended coordinates
    # ============================================================
    ix_src_ext = ix_src + n_abs
    iz_src_ext = iz_src + n_abs
    ix_receivers_ext = ix_receivers + n_abs
    iz_receivers_ext = iz_receivers + n_abs
    # ============================================================
    # 4. Check source and receivers
    # ============================================================ 
    if ix_src < 0 or ix_src >= nx:
        raise ValueError("Source x-position is outside physical model.")
    if iz_src < 0 or iz_src >= nz:
        raise ValueError("Source z-position is outside physical model.")
    for ir in range(n_receivers):
        if ix_receivers[ir] < 0 or ix_receivers[ir] >= nx:
            raise ValueError("Receiver x-position is outside physical model.")
        if iz_receivers[ir] < 0 or iz_receivers[ir] >= nz:
            raise ValueError("Receiver z-position is outside physical model.")
    # ===========================================================
    # 5. Buoyancy interpolation
    # rho is located at pressure nodes.
    # 1/rho must be interpolated to vx and vz positions.
    # ===========================================================
    inv_rho_vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float64)
    inv_rho_vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float64)
    # --------------------------------------------------------
    # 1/rho at vx
    # --------------------------------------------------------
    for iz in prange(nz_ext):
        inv_rho_vx[iz, 0] = 1.0 / rho_ext[iz, 0]
        for ix in range(1, nx_ext):
            inv_rho_vx[iz, ix] = 0.5 * (1.0 / rho_ext[iz, ix - 1] + 1.0 / rho_ext[iz, ix])
        inv_rho_vx[iz, nx_ext] = 1.0 / rho_ext[iz, nx_ext - 1]
    # --------------------------------------------------------
    # 1/rho at vz
    # --------------------------------------------------------
    for ix in prange(nx_ext):
        inv_rho_vz[0, ix] = 1.0 / rho_ext[0, ix]
        for iz in range(1, nz_ext):
            inv_rho_vz[iz, ix] = 0.5 * (1.0 / rho_ext[iz - 1, ix] + 1.0 / rho_ext[iz, ix])
        inv_rho_vz[nz_ext, ix] = 1.0 / rho_ext[nz_ext - 1, ix]
    # ============================================================
    # 6. Interpolate sigma to velocity grid
    # ============================================================
    sigma_x_vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float64)
    sigma_z_vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float64)
    # --------------------------------------------------------
    # sigma_x -> vx grid
    # --------------------------------------------------------
    for iz in prange(nz_ext):
        sigma_x_vx[iz, 0] = sigma_x[iz, 0]
        for ix in range(1, nx_ext):
            sigma_x_vx[iz, ix] = 0.5 * (sigma_x[iz, ix - 1]+sigma_x[iz, ix])
        sigma_x_vx[iz, nx_ext] = sigma_x[iz, nx_ext - 1]
    # --------------------------------------------------------
    # sigma_z -> vz grid
    # --------------------------------------------------------
    for ix in prange(nx_ext):
        sigma_z_vz[0, ix] = sigma_z[0, ix]
        for iz in range(1, nz_ext):
            sigma_z_vz[iz, ix] = 0.5 * (sigma_z[iz - 1, ix]+sigma_z[iz, ix])
        sigma_z_vz[nz_ext, ix] = sigma_z[nz_ext - 1, ix]
    # ============================================================
    # 7. PML coefficients
    # du/dt + sigma*u = F
    # u_new = a*u_old + b*F
    # a = (1 - sigma*dt/2)/(1 + sigma*dt/2)
    # b = dt/(1 + sigma*dt/2)
    # ============================================================
    a_vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float64)
    b_vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float64)
    a_vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float64)
    b_vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float64)
    a_px = np.zeros((nz_ext, nx_ext),dtype=np.float64)
    b_px = np.zeros((nz_ext, nx_ext),dtype=np.float64)
    a_pz = np.zeros((nz_ext, nx_ext),dtype=np.float64)
    b_pz = np.zeros((nz_ext, nx_ext),dtype=np.float64)
    # --------------------------------------------------------
    # vx coefficients
    # --------------------------------------------------------
    for iz in prange(nz_ext):
        for ix in range(nx_ext + 1):
            s = sigma_x_vx[iz, ix]
            den = 1.0 + 0.5 * dt * s
            a_vx[iz, ix] = (1.0 - 0.5 * dt * s) / den
            b_vx[iz, ix] = dt / den
    # --------------------------------------------------------
    # vz coefficients
    # --------------------------------------------------------
    for iz in prange(nz_ext + 1):
        for ix in range(nx_ext):
            s = sigma_z_vz[iz, ix]
            den = 1.0 + 0.5 * dt * s
            a_vz[iz, ix] = (1.0 - 0.5 * dt * s) / den
            b_vz[iz, ix] = dt / den
    # --------------------------------------------------------
    # px and pz coefficients
    # --------------------------------------------------------
    for iz in prange(nz_ext):
        for ix in range(nx_ext):
            sx = sigma_x[iz, ix]
            sz = sigma_z[iz, ix]
            den_x = 1.0 + 0.5 * dt * sx
            den_z = 1.0 + 0.5 * dt * sz
            a_px[iz, ix] = (1.0 - 0.5 * dt * sx) / den_x
            b_px[iz, ix] = dt / den_x
            a_pz[iz, ix] = (1.0 - 0.5 * dt * sz) / den_z
            b_pz[iz, ix] = dt / den_z
    # ========================================================
    # 8. Wavefield allocation
    # ========================================================
    # Total pressure
    p = np.zeros((nz_ext, nx_ext),dtype=np.float64)
    # Split pressure
    px = np.zeros((nz_ext, nx_ext),dtype=np.float64)
    pz = np.zeros((nz_ext, nx_ext),dtype=np.float64)
    # Particle velocity
    vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float64)
    vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float64)
    # ========================================================
    # 9. Output allocation
    # ========================================================
    seismogram_p = np.zeros((nt, n_receivers),dtype=np.float64)
    # Only save physical domain
    snapshots_p = np.zeros((n_snapshots, nz, nx),dtype=np.float64)
    snapshot_id = 0

    # ============================================================
    # 9. Time loop:
    # p^n
    #   ->
    # vx^(n+1/2), vz^(n+1/2)
    #   ->
    # px^(n+1), pz^(n+1)
    #   ->
    # p^(n+1)
    # ============================================================
    for it in range(nt-1):
        # Progress
        if (it + 1) % 100 == 0:
            print("Time step:",it + 1,"/",nt,"  Time:",(it + 1) * dt,"s")
        # -------------------------------------------------
        # 1). Update vx
        #
        # dvx/dt + sigma_x*vx
        #     = -(1/rho) dp/dx
        # -------------------------------------------------
        for iz in prange(N, nz_ext - N):
            for ix in range(N, nx_ext - N + 1):
                dpdx = 0.0
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dpdx += c * (p[iz, ix + m - 1] - p[iz, ix - m])
                dpdx /= dx
                vx[iz, ix] = a_vx[iz, ix] * vx[iz, ix] - b_vx[iz, ix] * inv_rho_vx[iz, ix] * dpdx
        # -------------------------------------------------
        # 2). Update vz
        # dvz/dt + sigma_z*vz
        #     = -(1/rho) dp/dz
        # -------------------------------------------------
        for iz in prange(N, nz_ext - N + 1):
            for ix in range(N, nx_ext - N):
                dpdz = 0.0
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dpdz += c * (p[iz + m - 1, ix]-p[iz - m, ix])
                dpdz /= dz
                vz[iz, ix] = a_vz[iz, ix]*vz[iz, ix]-b_vz[iz, ix]*inv_rho_vz[iz, ix]*dpdz
        # -------------------------------------------------
        # 3). Update split pressure
        # -------------------------------------------------
        for iz in prange(N, nz_ext - N):
            for ix in range(N, nx_ext - N):
                dvxdx = 0.0
                dvzdz = 0.0
                # --------------------------------------------
                # dvx/dx
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvxdx += c * (vx[iz, ix + m] - vx[iz, ix - m + 1] )
                dvxdx /= dx
                # --------------------------------------------
                # dvz/dz
                # --------------------------------------------
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvzdz += c * (vz[iz + m, ix] - vz[iz - m + 1, ix])
                dvzdz /= dz
                # --------------------------------------------
                # px
                # --------------------------------------------
                px[iz, ix] = a_px[iz, ix]*px[iz, ix]-b_px[iz, ix]*bulk_ext[iz, ix]*dvxdx
                # --------------------------------------------
                # pz
                # --------------------------------------------
                pz[iz, ix] = a_pz[iz, ix] * pz[iz, ix] - b_pz[iz, ix] * bulk_ext[iz, ix] * dvzdz
                # Total pressure
                p[iz, ix] = px[iz, ix] + pz[iz, ix]
        # -------------------------------------------------
        # 4). Add pressure source
        # total source: p += dt * wavelet[it]
        # split equally into px and pz
        # -------------------------------------------------
        source = 0.5 * dt * wavelet[it]
        px[iz_src_ext, ix_src_ext] += source
        pz[iz_src_ext, ix_src_ext] += source
        p[iz_src_ext, ix_src_ext] = px[iz_src_ext, ix_src_ext] + pz[iz_src_ext, ix_src_ext]
        # -------------------------------------------------
        # 5). Record receivers
        # seismogram_p[step] = p(step*dt)
        # -------------------------------------------------
        for ir in range(n_receivers):
            ix_rec_ext = ix_receivers_ext[ir]
            iz_rec_ext = iz_receivers_ext[ir]
            seismogram_p[it+1, ir] = p[iz_rec_ext,ix_rec_ext]
        # -------------------------------------------------
        # 6). Save physical-domain snapshot
        # -------------------------------------------------    
        if snapshot_id < n_snapshots:
            if (it+1) == snapshot_steps[snapshot_id]:
                for iz in prange(nz):
                    for ix in range(nx):
                        snapshots_p[snapshot_id,iz,ix] = p[iz + n_abs,ix + n_abs]
                snapshot_id += 1

    return seismogram_p,snapshots_p