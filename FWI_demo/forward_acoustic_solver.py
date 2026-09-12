import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def forward_acoustic_solver(
    velocity_ext,
    wavelet,
    coeff,
    ix_src,
    iz_src,
    ix_receiver,
    iz_receiver,
    dx,
    dz,
    dt,
    snapshot_steps,
    sigma,
    n_abs,
    store_wavefield_tt=True,
    store_velocity_sensitivity=False,
    verbose=True,
):

    # Extended computational domain
    nz_ext, nx_ext = velocity_ext.shape
    nt = len(wavelet)

    N = len(coeff)

    dx2 = dx * dx
    dz2 = dz * dz
    dt2 = dt * dt

    # Original physical model size
    nz = nz_ext - 2 * n_abs
    nx = nx_ext - 2 * n_abs

    # Source position in extended model
    ix_src_ext = ix_src + n_abs
    iz_src_ext = iz_src + n_abs

    # Receiver position in extended model
    ix_receiver_ext = ix_receiver + n_abs
    iz_receiver_ext = iz_receiver + n_abs
    n_receiver = len(ix_receiver_ext)

    # Wavefields defined on EXTENDED computational domai
    u_prev = np.zeros((nz_ext, nx_ext))
    u_curr = np.zeros((nz_ext, nx_ext))
    u_next = np.zeros((nz_ext, nx_ext))

    # Snapshots only store ORIGINAL physical domain
    # Number of snapshots
    nsnapshot = len(snapshot_steps)
    # Store snapshots
    snapshots = np.zeros((nsnapshot, nz, nx))
    isnapshot = 0

    # Store second-order derivative
    # 线搜索只需炮集；严格离散伴随则保存整个扩展域的局部速度导数。
    # 返回值第三项在 store_velocity_sensitivity=True 时为 d(u_next)/d(v_ext)。
    if store_velocity_sensitivity:
        wavefield_tt = np.zeros((nt - 1, nz_ext, nx_ext))
    elif store_wavefield_tt:
        wavefield_tt = np.zeros((nt, nz, nx))
    else:
        wavefield_tt = np.zeros((0, nz, nx))

    # shotgather
    shotgather = np.zeros((nt,n_receiver))

    # Time stepping
    for it in range(nt-1):
        # Progress
        if verbose and ((it + 1) % 100 == 0 or it == nt - 2):
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
                # 对实际离散更新式求导，sigma 在一次反演中保持不变。
                if store_velocity_sensitivity:
                    wavefield_tt[it, iz, ix] = 2.0*v*dt2*(d2u_dx2+d2u_dz2)/(1.0+damp*dt)
                
        # Source
        u_next[iz_src_ext, ix_src_ext] += velocity_ext[iz_src_ext, ix_src_ext]**2 * dt2 * wavelet[it]
        if store_velocity_sensitivity:
            wavefield_tt[it, iz_src_ext, ix_src_ext] += 2.0*velocity_ext[iz_src_ext, ix_src_ext]*dt2*wavelet[it]
        # ==================================================
        # Second-order time derivative
        #        u(n+1) - 2u(n) + u(n-1)
        # utt = --------------------------
        #                   dt^2
        # Only save the original physical model.
        # ==================================================
        for iz in prange(nz if store_wavefield_tt and not store_velocity_sensitivity else 0):
            iz_ext = iz + n_abs
            for ix in range(nx):
                ix_ext = ix + n_abs
                wavefield_tt[it, iz, ix] = (u_next[iz_ext, ix_ext] 
                                            - 2.0 * u_curr[iz_ext, ix_ext] 
                                            + u_prev[iz_ext, ix_ext]) / dt2        

        # Save shotgathers
        for ir in range(n_receiver):
            shotgather[it+1,ir] = u_next[iz_receiver_ext[ir], ix_receiver_ext[ir]]

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
    
    return snapshots,shotgather,wavefield_tt
