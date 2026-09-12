import numpy as np
from numba import njit, prange

@njit(parallel=True,fastmath=True)
def _acoustic_solver_dirichlet(
    velocity,
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
    #-----------------------------------------------
    # Basic parameters
    #----------------------------------------------- 
    nz, nx = velocity.shape
    nt = len(wavelet)
    t = np.arange(nt) * dt
    N = len(coeff) # FD order
    dx2 = dx * dx
    dz2 = dz * dz
    dt2 = dt * dt
    v2 = velocity[iz_src,ix_src]**2
    #-----------------------------------------------
    # Wavefields
    #-----------------------------------------------
    u_prev = np.zeros((nz, nx))
    u_curr = np.zeros((nz, nx))
    u_next = np.zeros((nz, nx))
    #-----------------------------------------------
    # Seismogram
    #-----------------------------------------------
    n_receivers = len(ix_receivers)
    seismogram = np.zeros((nt,n_receivers))
    #-----------------------------------------------
    # Snapshots
    #-----------------------------------------------
    n_snapshot = len(snapshot_steps)
    snapshots = np.zeros((n_snapshot, nz, nx))
    snapshot_id = 0
    #===============================================
    # Time looping
    #===============================================
    for it in range(nt-1):
        # Progress
        if (it + 1) % 100 == 0 or it == nt - 1:
            print("Time step:",it + 1,"/",nt,"  Time:",(it + 1) * dt,"s")
        #-----------------------------------------------
        # Dirichlet boundary
        #-----------------------------------------------
        u_next[:, :] = 0.0
        #-----------------------------------------------
        # Wavefield update, Kernel
        #-----------------------------------------------
        for iz in prange(N, nz - N):
            for ix in range(N, nx - N):
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
                v = velocity[iz, ix]
                u_next[iz, ix] = (
                    2.0 * u_curr[iz, ix]
                    -
                    u_prev[iz, ix]
                    +
                    v * v * dt2
                    * (d2u_dx2 + d2u_dz2)
                )
        #-----------------------------------------------
        # Source injection
        #-----------------------------------------------
        u_next[iz_src, ix_src] += v2 * dt2 * wavelet[it]
        #-----------------------------------------------
        # Seismogram
        #-----------------------------------------------
        for ir in range(n_receivers):
            ix_r = ix_receivers[ir]
            iz_r = iz_receivers[ir]
            seismogram[it+1,ir] = u_next[iz_r,ix_r]
        #-----------------------------------------------
        # Save snapshot
        #-----------------------------------------------
        if snapshot_id < n_snapshot:
            if (it+1) == snapshot_steps[snapshot_id]:
                snapshots[snapshot_id, :, :] = u_next
                snapshot_id += 1
        #-----------------------------------------------
        # Rotate wavefields
        #-----------------------------------------------
        temp = u_prev
        u_prev = u_curr
        u_curr = u_next
        u_next = temp       
    return seismogram,snapshots