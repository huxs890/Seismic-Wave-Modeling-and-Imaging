import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _acoustic_solver_PML(
    velocity_ext,
    dx,
    dz,
    dt,
    coeff,
    ix_src,
    iz_src,
    wavelet,
    ix_receivers,
    iz_receivers,
    snapshot_steps, # not _times
    sigma,
    n_abs,
):
    #-----------------------------------------------
    # Basic parameters
    #----------------------------------------------- 
    nz_ext, nx_ext = velocity_ext.shape # Extended computational domain
    nt = len(wavelet)
    t = np.arange(nt) * dt
    N = len(coeff)
    dx2 = dx * dx
    dz2 = dz * dz
    dt2 = dt * dt
    nz = nz_ext - 2 * n_abs # Original physical model size
    nx = nx_ext - 2 * n_abs
    #-----------------------------------------------
    # Source position in extended model
    #-----------------------------------------------
    ix_src_ext = ix_src + n_abs
    iz_src_ext = iz_src + n_abs
    v2 = velocity_ext[iz_src_ext, ix_src_ext]**2
    #-----------------------------------------------
    # Seismogram
    #-----------------------------------------------
    n_receivers = len(ix_receivers)
    seismogram = np.zeros((nt,n_receivers))    
    #-----------------------------------------------
    # Wavefields defined on extended computational domai
    #-----------------------------------------------
    u_prev = np.zeros((nz_ext, nx_ext))
    u_curr = np.zeros((nz_ext, nx_ext))
    u_next = np.zeros((nz_ext, nx_ext))
    #-----------------------------------------------
    # Snapshots only store original physical domain
    #-----------------------------------------------
    n_snapshot = len(snapshot_steps)  # Number of snapshots
    snapshots = np.zeros((n_snapshot, nz, nx))  # Store snapshots
    snapshot_id = 0
    #===============================================
    # Time loop
    #===============================================
    for it in range(nt-1):
        # Progress
        if (it + 1) % 100 == 0 or it == nt - 1:
            print("Time step:",it + 1,"/",nt,"  Time:",(it + 1) * dt,"s")
        #-----------------------------------------------
        # Reset wavefield
        #-----------------------------------------------
        u_next[:, :] = 0.0
        #-----------------------------------------------
        # Wavefield update
        #-----------------------------------------------
        for iz in prange(N, nz_ext - N):
            for ix in range(N, nx_ext - N):
                d2u_dx2 = 0.0
                d2u_dz2 = 0.0
                # x derivative
                for m in range(N):
                    d2u_dx2 += coeff[m] * (
                        u_curr[iz, ix + m + 1] 
                        + 
                        u_curr[iz, ix - m -1]
                        -
                        2.0 * u_curr[iz, ix])
                d2u_dx2 /= dx2
                # z derivative
                for m in range(N):
                    d2u_dz2 += coeff[m] * (
                         u_curr[iz + m + 1, ix] 
                         + 
                         u_curr[iz - m - 1, ix]
                         -
                         2.0 * u_curr[iz, ix])     
                d2u_dz2 /= dz2
                # time update
                v = velocity_ext[iz, ix]
                damp = sigma[iz, ix]
                u_next[iz, ix] = (2.0*u_curr[iz, ix]-(1.0-damp*dt)*u_prev[iz, ix]
                                  +v*v*dt2*(d2u_dx2+d2u_dz2)) / (1.0+damp*dt)
        #-----------------------------------------------
        # Source
        #-----------------------------------------------
        u_next[iz_src_ext, ix_src_ext] += v2 * dt2 * wavelet[it]
        #-----------------------------------------------
        # Seismogram
        #-----------------------------------------------
        for ir in range(n_receivers):
            ix_r_ext = ix_receivers[ir] + n_abs
            iz_r_ext = iz_receivers[ir] + n_abs
            seismogram[it+1,ir] = u_next[iz_r_ext,ix_r_ext]       
        #-----------------------------------------------------------------------     
        # Save snapshot, remove damping layers and store only original model
        #----------------------------------------------------------------------- 
        if snapshot_id < n_snapshot:
            if (it+1) == snapshot_steps[snapshot_id]:
                snapshots[snapshot_id, :, :] = u_next[n_abs:n_abs+nz, n_abs:n_abs+nx]
                snapshot_id += 1
        #--------------------------------------------------
        # Rotate wavefields
        #--------------------------------------------------
        temp = u_prev
        u_prev = u_curr
        u_curr = u_next
        u_next = temp
    return seismogram,snapshots