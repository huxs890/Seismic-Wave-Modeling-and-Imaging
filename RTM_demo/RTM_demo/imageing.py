import numpy as np
import matplotlib.pyplot as plt
from fd_coefficients import fd_coefficients
from forward_acoustic_solver import forward_acoustic_solver
from backward_acoustic_solver import backward_acoustic_solver
from boundary import damping_boundary

def RTM_imaging(
        velocity_smooth,
        dx,
        dz,
        dt,
        tmax,
        order,
        wavelet,
        ix_src,   # ix_src = [x1,x2,x3,...]
        iz_src,   # depth of source
        shotgather, # shotgather = [n_source,nt,nx]
        n_abs = 40,
):
    #=========================================================
    # Parameters
    #=========================================================
    nz, nx = velocity_smooth.shape
    velocity_smooth = np.asarray(velocity_smooth,dtype=np.float64)

    # Finite-difference order
    coeff = fd_coefficients(order)
    coeff = np.asarray(coeff,dtype=np.float64)

    # Snapshot times
    snapshot_times = np.arange(dt,tmax+dt,dt)
    snapshot_steps = np.rint(snapshot_times / dt).astype(np.int64)

    # =========================================================
    # Absorbing boundary
    # =========================================================
    velocity_ext, sigma = damping_boundary(
        velocity_smooth,
        dx,
        dz,
        n_abs,
        reflection=1e-6,
        power=2,
        )

    # =========================================================
    # Multi-shot RTM arrays
    # =========================================================
    n_src = len(ix_src)
    image1 = np.zeros((nz, nx),dtype=np.float64)
    illumination = np.zeros((nz, nx),dtype=np.float64)
    image2 = np.zeros((nz, nx),dtype=np.float64)

    #=========================================================
    # source steps
    #=========================================================
    for i in range(n_src):
        # forward modeling
        snapshots_forward = forward_acoustic_solver(
            velocity_ext,
            wavelet,
            coeff,
            ix_src[i],
            iz_src,
            dx,
            dz,
            dt,
            snapshot_steps,
            sigma,
            n_abs,
            )[0]

        # backward modeling
        snapshots_backward = backward_acoustic_solver(
            velocity_ext,
            coeff,
            dx,
            dz,
            dt,
            shotgather[i],
            iz_src,
            snapshot_steps,
            sigma,
            n_abs,
            )
        #==========================================================
        # RTM imaging condition
        #==========================================================
        cross_corr = np.zeros((snapshots_forward.shape[1], snapshots_forward.shape[2]), dtype=np.float64)
        source_energy = np.zeros((snapshots_forward.shape[1], snapshots_forward.shape[2]), dtype=np.float64)
        nt = len(snapshot_times)
        for it in range(nt):
            reverse_it = nt - 1 - it
            pf = snapshots_forward[reverse_it]
            pb = snapshots_backward[it]

            # cross correlation
            cross_corr += pf * pb 

            # Source illumination
            source_energy += pf * pf 

        # 1.Cross-correlation image
        image_shot1 = cross_corr # single-shot normalized image
        image1 += image_shot1 # multi-shot stacking  

        # 2.Source-normalized image
        illumination += source_energy 
    # 2.Source-normalized image
    eps = 1e-5 * np.max(illumination)
    image2 = image1 / (illumination + eps) # multi-shot stacking   

    return image1,image2 