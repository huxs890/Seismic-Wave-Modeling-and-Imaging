import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def backward_acoustic_solver(
    velocity_ext,
    coeff,
    dx,
    dz,
    dt,
    shot_gather,
    iz_receiver,
    snapshot_steps,
    sigma,
    n_abs,
):

    # Extended computational domain
    nz_ext, nx_ext = velocity_ext.shape
    nt = shot_gather.shape[0]
    print(nz_ext,nx_ext,nt)

    N = len(coeff)

    dx2 = dx * dx
    dz2 = dz * dz
    dt2 = dt * dt

    # Original physical model size
    nz = nz_ext - 2 * n_abs
    nx = nx_ext - 2 * n_abs

    # Wavefields defined on EXTENDED computational domai
    u_prev = np.zeros((nz_ext, nx_ext))
    u_curr = np.zeros((nz_ext, nx_ext))
    u_next = np.zeros((nz_ext, nx_ext))
    iz_src_ext = iz_receiver + n_abs

    # Snapshots only store ORIGINAL physical domain
    # Number of snapshots
    nsnapshot = len(snapshot_steps)
    # Store snapshots
    snapshots = np.zeros((nsnapshot, nz, nx))
    isnapshot = 0

    # Time stepping
    for it in range(nt):
        # Progress
        if (it + 1) % 100 == 0 or it == nt - 1:
            print("Time step:",it + 1,"/",nt,"  Time:",(it + 1) * dt,"s")
            
        # Reset wavefield
        u_next[:, :] = 0.0
        # Wavefield update
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
                
        # Source
        u_next[iz_src_ext, n_abs:n_abs+nx] += shot_gather[nt-it-1,:]

        # Save snapshot, remove damping layers and store only original model
        if isnapshot < nsnapshot:
            if (it+1) == snapshot_steps[isnapshot]:
                snapshots[isnapshot, :, :] = u_next[n_abs:n_abs+nz, n_abs:n_abs+nx]
                isnapshot += 1

        # Rotate wavefields
        temp = u_prev
        u_prev = u_curr
        u_curr = u_next
        u_next = temp
    
    return snapshots