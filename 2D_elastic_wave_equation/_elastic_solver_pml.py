import numpy as np
from numba import njit, prange
# ============================================================
# 2-D VTI elastic wave equation
# First-order stress-velocity formulation
# Split-field PML boundary
# ------------------------------------------------------------
# Governing equations
# ------------------------------------------------------------
# rho * dvx/dt = dsxx/dx + dsxz/dz
# rho * dvz/dt = dsxz/dx + dszz/dz
# dsxx/dt = C11 * dvx/dx + C13 * dvz/dz
# dszz/dt = C13 * dvx/dx + C33 * dvz/dz
# dsxz/dt = C55 * (dvx/dz + dvz/dx)
# ------------------------------------------------------------
# Split-field PML
# ------------------------------------------------------------
# vx = vx_x + vx_z
# vz = vz_x + vz_z
# sxx = sxx_x + sxx_z
# szz = szz_x + szz_z
# sxz = sxz_x + sxz_z
# x-direction terms -> sigma_x
# z-direction terms -> sigma_z
# ------------------------------------------------------------
# Staggered grid
# ------------------------------------------------------------
# sxx, szz : (nz_ext, nx_ext)
# vx       : (nz_ext, nx_ext + 1)
# vz       : (nz_ext + 1, nx_ext)
# sxz      : (nz_ext + 1, nx_ext + 1)
#
#              sxz -------- vz -------- sxz
#               |                        |
#               |                        |
#               vx      sxx,szz          vx
#               |                        |
#               |                        |
#              sxz -------- vz -------- sxz
#
# rho_ext, Cij_ext and sigma_x/sigma_z are defined
# on the normal-stress grid.
# Source and receiver indices are defined in the
# original physical domain and shifted internally
# by n_abs.
# ============================================================
@njit(parallel=True, fastmath=True)
def _elastic_vti_solver_split_PML(
    rho_ext,
    c11_ext,
    c13_ext,
    c33_ext,
    c55_ext,
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
    # ========================================================
    # 1. Basic parameters
    # ========================================================
    nz_ext, nx_ext = rho_ext.shape
    # Physical model size
    nz = nz_ext - 2 * n_abs
    nx = nx_ext - 2 * n_abs
    nt = len(wavelet)
    # Half stencil width
    N = len(coeff)
    n_receivers = len(ix_receivers)
    n_snapshots = len(snapshot_steps)
    # ========================================================
    # 2. Physical coordinates -> extended coordinates
    # ========================================================
    ix_src_ext = ix_src + n_abs
    iz_src_ext = iz_src + n_abs
    ix_receivers_ext = ix_receivers + n_abs
    iz_receivers_ext = iz_receivers + n_abs
    # ========================================================
    # 3. Buoyancy interpolation
    # rho is defined at sxx/szz locations.
    # 1/rho is interpolated to vx and vz locations.
    # ========================================================
    inv_rho_vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float32)
    inv_rho_vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float32)
    # --------------------------------------------------------
    # 1/rho -> vx
    # --------------------------------------------------------
    for iz in prange(nz_ext):
        inv_rho_vx[iz, 0] = 1.0 / rho_ext[iz, 0]
        for ix in range(1, nx_ext):
            inv_rho_vx[iz, ix] = 0.5 * (1.0 / rho_ext[iz, ix - 1]+1.0 / rho_ext[iz, ix])
        inv_rho_vx[iz, nx_ext] = 1.0 / rho_ext[iz, nx_ext - 1]
    # --------------------------------------------------------
    # 1/rho -> vz
    # --------------------------------------------------------
    for ix in prange(nx_ext):
        inv_rho_vz[0, ix] = 1.0 / rho_ext[0, ix]
        for iz in range(1, nz_ext):
            inv_rho_vz[iz, ix] = 0.5 * (1.0 / rho_ext[iz - 1, ix]+1.0 / rho_ext[iz, ix])
        inv_rho_vz[nz_ext, ix] = 1.0 / rho_ext[nz_ext - 1, ix]
    # ========================================================
    # 4. C55 interpolation -> sxz grid
    # sxz is located half a grid in both x and z.
    # ========================================================
    c55_xz = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    for iz in prange(1, nz_ext):
        for ix in range(1, nx_ext):
            c1 = c55_ext[iz - 1, ix - 1]
            c2 = c55_ext[iz - 1, ix]
            c3 = c55_ext[iz, ix - 1]
            c4 = c55_ext[iz, ix]
            c55_xz[iz, ix] = 4/(1/c1 + 1/c2 + 1/c3 + 1/c4)
    # ========================================================
    # 5. Interpolate sigma fields to staggered grids
    # ========================================================
    # --------------------------------------------------------
    # sigma_x / sigma_z at vx positions
    # --------------------------------------------------------
    sigma_x_vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float32)
    sigma_z_vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float32)
    for iz in prange(nz_ext):
        sigma_x_vx[iz, 0] = sigma_x[iz, 0]
        sigma_z_vx[iz, 0] = sigma_z[iz, 0]
        for ix in range(1, nx_ext):
            sigma_x_vx[iz, ix] = 0.5 * (sigma_x[iz, ix - 1]+sigma_x[iz, ix])
            sigma_z_vx[iz, ix] = 0.5 * (sigma_z[iz, ix - 1]+sigma_z[iz, ix])
        sigma_x_vx[iz, nx_ext] = sigma_x[iz, nx_ext - 1]
        sigma_z_vx[iz, nx_ext] = sigma_z[iz, nx_ext - 1]
    # --------------------------------------------------------
    # sigma_x / sigma_z at vz positions
    # --------------------------------------------------------
    sigma_x_vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float32)
    sigma_z_vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float32)
    for ix in prange(nx_ext):
        sigma_x_vz[0, ix] = sigma_x[0, ix]
        sigma_z_vz[0, ix] = sigma_z[0, ix]
        for iz in range(1, nz_ext):
            sigma_x_vz[iz, ix] = 0.5 * (sigma_x[iz - 1, ix]+sigma_x[iz, ix])
            sigma_z_vz[iz, ix] = 0.5 * (sigma_z[iz - 1, ix]+sigma_z[iz, ix])
        sigma_x_vz[nz_ext, ix] = sigma_x[nz_ext - 1, ix]
        sigma_z_vz[nz_ext, ix] = sigma_z[nz_ext - 1, ix]
    # --------------------------------------------------------
    # sigma_x / sigma_z at sxz positions
    # --------------------------------------------------------
    sigma_x_xz = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    sigma_z_xz = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    for iz in prange(1, nz_ext):
        for ix in range(1, nx_ext):
            sigma_x_xz[iz, ix] = 0.25 * (sigma_x[iz - 1, ix - 1]+sigma_x[iz - 1, ix]+sigma_x[iz, ix - 1]+sigma_x[iz, ix])
            sigma_z_xz[iz, ix] = 0.25 * (sigma_z[iz - 1, ix - 1]+sigma_z[iz - 1, ix]+sigma_z[iz, ix - 1]+sigma_z[iz, ix])
    # ========================================================
    # 6. PML coefficient helper relation
    # du/dt + sigma*u = F
    # discretized as:
    # u_new = a*u_old + b*F
    # where
    #     (1 - sigma*dt/2)
    # a = -----------------
    #     (1 + sigma*dt/2)
    #            dt
    # b = -----------------
    #     (1 + sigma*dt/2)
    # sigma = 0:
    # a = 1
    # b = dt
    # ========================================================
    a_vxx = np.zeros_like(inv_rho_vx)
    b_vxx = np.zeros_like(inv_rho_vx)
    a_vxz = np.zeros_like(inv_rho_vx)
    b_vxz = np.zeros_like(inv_rho_vx)
    a_vzx = np.zeros_like(inv_rho_vz)
    b_vzx = np.zeros_like(inv_rho_vz)
    a_vzz = np.zeros_like(inv_rho_vz)
    b_vzz = np.zeros_like(inv_rho_vz)    
    # ---------------------------------------------
    # 1) Velocity PML coefficients
    # ---------------------------------------------
    # vx x- and z-split parts
    for iz in prange(nz_ext):
        for ix in range(nx_ext + 1):
            sx = sigma_x_vx[iz, ix]
            sz = sigma_z_vx[iz, ix]
            den_x = 1.0 + 0.5 * dt * sx
            den_z = 1.0 + 0.5 * dt * sz
            a_vxx[iz, ix] = (1.0 - 0.5 * dt * sx) / den_x
            b_vxx[iz, ix] = dt / den_x
            a_vxz[iz, ix] = (1.0 - 0.5 * dt * sz) / den_z
            b_vxz[iz, ix] = dt / den_z
    # vz x- and z-split parts
    for iz in prange(nz_ext + 1):
        for ix in range(nx_ext):
            sx = sigma_x_vz[iz, ix]
            sz = sigma_z_vz[iz, ix]
            den_x = 1.0 + 0.5 * dt * sx
            den_z = 1.0 + 0.5 * dt * sz
            a_vzx[iz, ix] = (1.0 - 0.5 * dt * sx) / den_x
            b_vzx[iz, ix] = dt / den_x
            a_vzz[iz, ix] = (1.0 - 0.5 * dt * sz) / den_z
            b_vzz[iz, ix] = dt / den_z
    # ---------------------------------------------
    # 2) Normal-stress PML coefficients
    # sxx/szz are defined at integer grid points.
    # ---------------------------------------------
    a_sx = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    b_sx = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    a_sz = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    b_sz = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    for iz in prange(nz_ext):
        for ix in range(nx_ext):
            sx = sigma_x[iz, ix]
            sz = sigma_z[iz, ix]
            den_x = 1.0 + 0.5 * dt * sx
            den_z = 1.0 + 0.5 * dt * sz
            a_sx[iz, ix] = (1.0 - 0.5 * dt * sx) / den_x
            b_sx[iz, ix] = dt / den_x
            a_sz[iz, ix] = (1.0 - 0.5 * dt * sz) / den_z
            b_sz[iz, ix] = dt / den_z
    # ---------------------------------------------
    # 3) Shear-stress PML coefficients
    # ---------------------------------------------
    a_sxz_x = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    b_sxz_x = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    a_sxz_z = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    b_sxz_z = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    for iz in prange(1, nz_ext):
        for ix in range(1, nx_ext):
            sx = sigma_x_xz[iz, ix]
            sz = sigma_z_xz[iz, ix]
            den_x = 1.0 + 0.5 * dt * sx
            den_z = 1.0 + 0.5 * dt * sz
            a_sxz_x[iz, ix] = (1.0 - 0.5 * dt * sx) / den_x
            b_sxz_x[iz, ix] = dt / den_x
            a_sxz_z[iz, ix] = (1.0 - 0.5 * dt * sz) / den_z
            b_sxz_z[iz, ix] = dt / den_z
    # ========================================================
    # 7. Split velocity fields
    # ========================================================
    vx_x = np.zeros((nz_ext, nx_ext + 1),dtype=np.float32)
    vx_z = np.zeros((nz_ext, nx_ext + 1),dtype=np.float32)
    vz_x = np.zeros((nz_ext + 1, nx_ext),dtype=np.float32)
    vz_z = np.zeros((nz_ext + 1, nx_ext),dtype=np.float32)
    # Total velocity
    vx = np.zeros((nz_ext, nx_ext + 1),dtype=np.float32)
    vz = np.zeros((nz_ext + 1, nx_ext),dtype=np.float32)
    vx_old = np.zeros((nz_ext, nx_ext + 1),dtype=np.float32)
    vz_old = np.zeros((nz_ext + 1, nx_ext),dtype=np.float32)
    # ========================================================
    # 8. Split normal stresses
    # ========================================================
    sxx_x = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    sxx_z = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    szz_x = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    szz_z = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    sxx = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    szz = np.zeros((nz_ext, nx_ext),dtype=np.float32)
    # ========================================================
    # 9. Split shear stress
    # ========================================================
    sxz_x = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    sxz_z = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    sxz = np.zeros((nz_ext + 1, nx_ext + 1),dtype=np.float32)
    # ========================================================
    # 10. Output allocation
    # ========================================================
    seismogram_vx = np.zeros((nt, n_receivers),dtype=np.float32)
    seismogram_vz = np.zeros((nt, n_receivers),dtype=np.float32)
    snapshots_vx = np.zeros((n_snapshots, nz, nx),dtype=np.float32)
    snapshots_vz = np.zeros((n_snapshots, nz, nx),dtype=np.float32)
    snapshot_id = 0
    # ========================================================
    # 11. Time Loop
    #
    # stress^n
    #       ->
    # split velocity^(n+1/2)
    #       ->
    # total velocity^(n+1/2)
    #       ->
    # split stress^(n+1)
    #       ->
    # total stress^(n+1)
    #
    # ========================================================
    for it in range(nt - 1):
        # ====================================================
        # 1). Update vx split fields
        # vx_x:
        # dvx_x/dt + sigma_x vx_x =
        # (1/rho) dsxx/dx
        #
        # vx_z:
        # dvx_z/dt + sigma_z vx_z
        #     =
        # (1/rho) dsxz/dz
        # ====================================================
        for iz in prange(N, nz_ext - N):
            for ix in range(N, nx_ext - N + 1):
                dsxx_dx = np.float32(0.0)
                dsxz_dz = np.float32(0.0)
                # dsxx/dx
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dsxx_dx += c * (sxx[iz, ix + m - 1]-sxx[iz, ix - m])
                dsxx_dx /= dx
                # dsxz/dz
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dsxz_dz += c * (sxz[iz + m, ix]-sxz[iz - m + 1, ix])
                dsxz_dz /= dz
                vx_x[iz, ix] = a_vxx[iz, ix]*vx_x[iz, ix]+b_vxx[iz, ix]*inv_rho_vx[iz, ix]*dsxx_dx
                vx_z[iz, ix] = a_vxz[iz, ix]*vx_z[iz, ix]+b_vxz[iz, ix]*inv_rho_vx[iz, ix]*dsxz_dz
                vx[iz, ix] = vx_x[iz, ix]+vx_z[iz, ix]
        # ====================================================
        # 2). Update vz split fields
        # vz_x:
        # dvz_x/dt + sigma_x vz_x
        #     =
        # (1/rho) dsxz/dx
        #
        # vz_z:
        # dvz_z/dt + sigma_z vz_z
        #     =
        # (1/rho) dszz/dz
        # ====================================================
        for iz in prange(N, nz_ext - N + 1):
            for ix in range(N, nx_ext - N):
                dsxz_dx = np.float32(0.0)
                dszz_dz = np.float32(0.0)
                # dsxz/dx
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dsxz_dx += c * (sxz[iz, ix + m]-sxz[iz, ix - m + 1])
                dsxz_dx /= dx
                # dszz/dz
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dszz_dz += c * (szz[iz + m - 1, ix]-szz[iz - m, ix])
                dszz_dz /= dz

                vz_x[iz, ix] = a_vzx[iz, ix]*vz_x[iz, ix] + b_vzx[iz, ix] * inv_rho_vz[iz, ix] * dsxz_dx
                vz_z[iz, ix] = a_vzz[iz, ix] * vz_z[iz, ix] + b_vzz[iz, ix]* inv_rho_vz[iz, ix] * dszz_dz
                vz[iz, ix] = vz_x[iz, ix]+vz_z[iz, ix]
        # ====================================================
        # 3). Update split normal stresses
        # sxx_x:
        # dsxx_x/dt + sigma_x sxx_x
        #     = C11 dvx/dx
        #
        # sxx_z:
        # dsxx_z/dt + sigma_z sxx_z
        #     = C13 dvz/dz
        #
        # szz_x:
        # dszz_x/dt + sigma_x szz_x
        #     = C13 dvx/dx
        #
        # szz_z:
        # dszz_z/dt + sigma_z szz_z
        #     = C33 dvz/dz
        # ====================================================
        for iz in prange(N, nz_ext - N):
            for ix in range(N, nx_ext - N):
                dvx_dx = np.float32(0.0)
                dvz_dz = np.float32(0.0)
                # dvx/dx
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvx_dx += c * (vx[iz, ix + m] - vx[iz, ix - m + 1])
                dvx_dx /= dx
                # dvz/dz
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvz_dz += c * (vz[iz + m, ix]-vz[iz - m + 1, ix])
                dvz_dz /= dz
                # sxx split fields
                sxx_x[iz, ix] = a_sx[iz, ix] * sxx_x[iz, ix] + b_sx[iz, ix] * c11_ext[iz, ix] * dvx_dx
                sxx_z[iz, ix] = a_sz[iz, ix] * sxx_z[iz, ix] + b_sz[iz, ix] * c13_ext[iz, ix] * dvz_dz
                # szz split fields
                szz_x[iz, ix] = a_sx[iz, ix] * szz_x[iz, ix] + b_sx[iz, ix] * c13_ext[iz, ix] * dvx_dx
                szz_z[iz, ix] = a_sz[iz, ix] * szz_z[iz, ix] + b_sz[iz, ix] * c33_ext[iz, ix] * dvz_dz
                # Total stresses
                sxx[iz, ix] = sxx_x[iz, ix] + sxx_z[iz, ix]
                szz[iz, ix] = szz_x[iz, ix] + szz_z[iz, ix]
        # ====================================================
        # 4). Update split shear stress
        # sxz_x:
        # dsxz_x/dt + sigma_x sxz_x
        #     =
        # C55 dvz/dx
        #
        # sxz_z:
        # dsxz_z/dt + sigma_z sxz_z
        #     =
        # C55 dvx/dz
        # ====================================================
        for iz in prange(N, nz_ext - N + 1):
            for ix in range(N, nx_ext - N + 1):
                dvx_dz = np.float32(0.0)
                dvz_dx = np.float32(0.0)
                # dvx/dz
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvx_dz += c * (vx[iz + m - 1, ix]-vx[iz - m, ix])
                dvx_dz /= dz
                # dvz/dx
                for m in range(1, N + 1):
                    c = coeff[m - 1]
                    dvz_dx += c * (vz[iz, ix + m - 1]-vz[iz, ix - m])
                dvz_dx /= dx
                sxz_x[iz, ix] = a_sxz_x[iz, ix]*sxz_x[iz, ix]+ b_sxz_x[iz, ix] * c55_xz[iz, ix] * dvz_dx
                sxz_z[iz, ix] = a_sxz_z[iz, ix]*sxz_z[iz, ix]+ b_sxz_z[iz, ix] * c55_xz[iz, ix] * dvx_dz
                sxz[iz, ix] = sxz_x[iz, ix] + sxz_z[iz, ix]
        # ====================================================
        # 5). Explosive stress source
        # Total desired source:
        # sxx += dt * wavelet
        # szz += dt * wavelet
        # Each normal stress is split equally between
        # x and z PML components.
        # ====================================================
        source = 0.5 * dt  * wavelet[it]
        # sxx source
        sxx_x[iz_src_ext,ix_src_ext] += source
        sxx_z[iz_src_ext,ix_src_ext] += source
        # szz source
        szz_x[iz_src_ext,ix_src_ext] += source
        szz_z[iz_src_ext,ix_src_ext] += source
        # Reconstruct total stress at source
        sxx[iz_src_ext,ix_src_ext] = sxx_x[iz_src_ext, ix_src_ext]+sxx_z[iz_src_ext, ix_src_ext]
        szz[iz_src_ext,ix_src_ext] = szz_x[iz_src_ext, ix_src_ext]+ szz_z[iz_src_ext, ix_src_ext]
        # ====================================================
        # Current physical time step
        # ====================================================
        step = it
        # ====================================================
        # 6). Record receivers
        # vx and vz live on half grids.
        # Interpolate them back to the normal-stress grid
        # before recording.
        #
        # Time interpolation:
        # v^n = 0.5 * (v^(n-1/2)+v^(n+1/2))
        # ====================================================
        for ir in range(n_receivers):
            ix_rec = ix_receivers_ext[ir]
            iz_rec = iz_receivers_ext[ir]
            vx_time = 0.5 * (vx_old[iz_rec, ix_rec]+vx[iz_rec, ix_rec])
            vx_time_right = 0.5 * (vx_old[iz_rec, ix_rec + 1]+vx[iz_rec, ix_rec + 1])
            vz_time = 0.5* (vz_old[iz_rec, ix_rec]+vz[iz_rec, ix_rec])
            vz_time_down = 0.5 * (vz_old[iz_rec + 1, ix_rec]+vz[iz_rec + 1, ix_rec])
            seismogram_vx[step, ir] = 0.5*(vx_time + vx_time_right)
            seismogram_vz[step, ir] = 0.5*(vz_time + vz_time_down)
        # ====================================================
        # 7). Save physical-domain snapshots
        # Output does not contain PML region.
        # ====================================================
        if snapshot_id < n_snapshots:
            if step == snapshot_steps[snapshot_id]:
                for iz in prange(nz):
                    iz_ext = iz + n_abs
                    for ix in range(nx):
                        ix_ext = ix + n_abs

                        vx_left = 0.5 * (vx_old[iz_ext, ix_ext] + vx[iz_ext, ix_ext])
                        vx_right = 0.5 * (vx_old[iz_ext, ix_ext + 1] + vx[iz_ext, ix_ext + 1])
                        snapshots_vx[snapshot_id,iz,ix] = 0.5 * (vx_left+vx_right)

                        vz_up = 0.5 * (vz_old[iz_ext, ix_ext]+vz[iz_ext, ix_ext])
                        vz_down = 0.5 * (vz_old[iz_ext + 1, ix_ext]+vz[iz_ext + 1, ix_ext])
                        snapshots_vz[snapshot_id,iz,ix] = np.float32(0.5) * (vz_up+vz_down)
                snapshot_id += 1
        # -----------------------------------------
        # 8). Save current half-time velocity for next iteration
        # -----------------------------------------
        vx_old[:, :] = vx[:, :]
        vz_old[:, :] = vz[:, :]
        # ====================================================
        # 9). Progress
        # ====================================================
        if step % 100 == 0 or step == nt - 1:
            print("Time step:",step,"/",nt - 1," Time:",step * dt,"s")

    # ========================================================
    # Output
    # ========================================================
    return seismogram_vx,seismogram_vz,snapshots_vx,snapshots_vz,