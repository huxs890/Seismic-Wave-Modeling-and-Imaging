"""High-order 2-D isotropic elastic-wave solver on a staggered grid.

The five wavefield variables use the Virieux layout::

    sxx, szz : (z, x)                 shape (nz, nx)
    vx       : (z, x + 1/2)           shape (nz, nx + 1)
    vz       : (z + 1/2, x)           shape (nz + 1, nx)
    sxz      : (z + 1/2, x + 1/2)     shape (nz + 1, nx + 1)

Stress and velocity are also staggered by half a time step.  The outer N grid
layers are not updated, which imposes a reflective homogeneous Dirichlet
boundary.  Use an absorbing boundary for production simulations.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # Small correctness tests can still run without Numba.
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator


def staggered_fd_coefficients(order: int) -> np.ndarray:
    """Return coefficients for a staggered-grid first derivative.

    Parameters
    ----------
    order : int
        Even spatial accuracy order: 2, 4, 6, ...

    Returns
    -------
    numpy.ndarray
        Coefficients ``[c1, ..., cN]``, where ``N = order // 2``.

    Notes
    -----
    The derivative at x is approximated by

    ``df/dx = sum(c[m] * (f(x+a_m*h)-f(x-a_m*h))) / h``,

    where ``a_m = m - 1/2``.  These are the same coefficients used by a
    first-order pressure-velocity acoustic staggered-grid solver.
    """
    if order <= 0 or order % 2 != 0:
        raise ValueError("order must be a positive even integer")

    n = order // 2
    matrix = np.empty((n, n), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    rhs[0] = 1.0

    for q in range(1, n + 1):
        power = 2 * q - 1
        for m in range(1, n + 1):
            a = m - 0.5
            matrix[q - 1, m - 1] = 2.0 * a**power / math.factorial(power)

    return np.linalg.solve(matrix, rhs)


def ricker_wavelet(f0: float, dt: float, nt: int, amplitude: float = 1.0) -> np.ndarray:
    """Create a Ricker wavelet whose peak occurs at ``1.5 / f0``."""
    if f0 <= 0.0 or dt <= 0.0 or nt <= 0:
        raise ValueError("f0, dt, and nt must be positive")
    t = np.arange(nt, dtype=np.float64) * dt
    tau = t - 1.5 / f0
    a = (np.pi * f0 * tau) ** 2
    return amplitude * (1.0 - 2.0 * a) * np.exp(-a)


def elastic_parameters(
    vp: np.ndarray, vs: np.ndarray, rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``vp``, ``vs``, and density to the Lame parameters."""
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if vp.shape != vs.shape or vp.shape != rho.shape or vp.ndim != 2:
        raise ValueError("vp, vs, and rho must be 2-D arrays with equal shapes")
    if np.any(vp <= 0.0) or np.any(vs <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("vp, vs, and rho must be strictly positive")
    if np.any(vp * vp <= 2.0 * vs * vs):
        raise ValueError("vp**2 must be greater than 2*vs**2 so lambda is positive")
    mu = rho * vs**2
    lam = rho * vp**2 - 2.0 * mu
    return lam, mu


@njit(cache=True)
def _stagger_material_parameters(rho, mu):
    """Interpolate buoyancy to velocity points and mu to shear-stress points."""
    nz, nx = rho.shape
    inv_rho_vx = np.zeros((nz, nx + 1), dtype=np.float64)
    inv_rho_vz = np.zeros((nz + 1, nx), dtype=np.float64)
    mu_xz = np.zeros((nz + 1, nx + 1), dtype=np.float64)

    # Arithmetic averaging of buoyancy is equivalent to harmonic averaging rho.
    for iz in range(nz):
        inv_rho_vx[iz, 0] = 1.0 / rho[iz, 0]
        inv_rho_vx[iz, nx] = 1.0 / rho[iz, nx - 1]
        for ix in range(1, nx):
            inv_rho_vx[iz, ix] = 0.5 * (
                1.0 / rho[iz, ix - 1] + 1.0 / rho[iz, ix]
            )

    for ix in range(nx):
        inv_rho_vz[0, ix] = 1.0 / rho[0, ix]
        inv_rho_vz[nz, ix] = 1.0 / rho[nz - 1, ix]
        for iz in range(1, nz):
            inv_rho_vz[iz, ix] = 0.5 * (
                1.0 / rho[iz - 1, ix] + 1.0 / rho[iz, ix]
            )

    # Four-point harmonic average is robust at elastic material interfaces.
    for iz in range(1, nz):
        for ix in range(1, nx):
            mu_xz[iz, ix] = 4.0 / (
                1.0 / mu[iz - 1, ix - 1]
                + 1.0 / mu[iz - 1, ix]
                + 1.0 / mu[iz, ix - 1]
                + 1.0 / mu[iz, ix]
            )

    return inv_rho_vx, inv_rho_vz, mu_xz


@njit(parallel=True, fastmath=True, cache=True)
def _elastic_solver_dirichlet_core(
    wavelet,
    coeff,
    ix_src,
    iz_src,
    dx,
    dz,
    dt,
    rho,
    lam,
    mu,
    snapshot_steps,
):
    """Numba-compiled implementation; call ``elastic_solver_dirichlet``."""
    nz, nx = rho.shape
    nt = len(wavelet)
    n = len(coeff)
    ns = len(snapshot_steps)

    inv_rho_vx, inv_rho_vz, mu_xz = _stagger_material_parameters(rho, mu)

    sxx = np.zeros((nz, nx), dtype=np.float64)
    szz = np.zeros((nz, nx), dtype=np.float64)
    vx = np.zeros((nz, nx + 1), dtype=np.float64)
    vz = np.zeros((nz + 1, nx), dtype=np.float64)
    sxz = np.zeros((nz + 1, nx + 1), dtype=np.float64)

    snapshots_sxx = np.zeros((ns, nz, nx), dtype=np.float64)
    snapshots_szz = np.zeros((ns, nz, nx), dtype=np.float64)
    snapshots_vx = np.zeros((ns, nz, nx + 1), dtype=np.float64)
    snapshots_vz = np.zeros((ns, nz + 1, nx), dtype=np.float64)
    snapshots_sxz = np.zeros((ns, nz + 1, nx + 1), dtype=np.float64)
    snapshot_id = 0

    for it in range(nt):

        # Progress
        if (it + 1) % 100 == 0 or it == nt - 1:
            print("Time step:",it + 1,"/",nt,"  Time:",(it + 1) * dt,"s")       

        # 1. Velocity at time n+1/2 from stress at time n.
        for iz in prange(n, nz - n):
            for ix in range(n, nx - n + 1):
                dsxx_dx = 0.0
                dsxz_dz = 0.0
                for m in range(1, n + 1):
                    c = coeff[m - 1]
                    dsxx_dx += c * (sxx[iz, ix + m - 1] - sxx[iz, ix - m])
                    dsxz_dz += c * (sxz[iz + m, ix] - sxz[iz - m + 1, ix])
                vx[iz, ix] += dt * inv_rho_vx[iz, ix] * (
                    dsxx_dx / dx + dsxz_dz / dz
                )

        for iz in prange(n, nz - n + 1):
            for ix in range(n, nx - n):
                dsxz_dx = 0.0
                dszz_dz = 0.0
                for m in range(1, n + 1):
                    c = coeff[m - 1]
                    dsxz_dx += c * (sxz[iz, ix + m] - sxz[iz, ix - m + 1])
                    dszz_dz += c * (szz[iz + m - 1, ix] - szz[iz - m, ix])
                vz[iz, ix] += dt * inv_rho_vz[iz, ix] * (
                    dsxz_dx / dx + dszz_dz / dz
                )

        # 2. Normal stresses at time n+1 from velocity at time n+1/2.
        for iz in prange(n, nz - n):
            for ix in range(n, nx - n):
                dvx_dx = 0.0
                dvz_dz = 0.0
                for m in range(1, n + 1):
                    c = coeff[m - 1]
                    dvx_dx += c * (vx[iz, ix + m] - vx[iz, ix - m + 1])
                    dvz_dz += c * (vz[iz + m, ix] - vz[iz - m + 1, ix])
                dvx_dx /= dx
                dvz_dz /= dz
                sxx[iz, ix] += dt * (
                    (lam[iz, ix] + 2.0 * mu[iz, ix]) * dvx_dx
                    + lam[iz, ix] * dvz_dz
                )
                szz[iz, ix] += dt * (
                    lam[iz, ix] * dvx_dx
                    + (lam[iz, ix] + 2.0 * mu[iz, ix]) * dvz_dz
                )

        # 3. Shear stress at the cell corners.
        for iz in prange(n, nz - n + 1):
            for ix in range(n, nx - n + 1):
                dvx_dz = 0.0
                dvz_dx = 0.0
                for m in range(1, n + 1):
                    c = coeff[m - 1]
                    dvx_dz += c * (vx[iz + m - 1, ix] - vx[iz - m, ix])
                    dvz_dx += c * (vz[iz, ix + m - 1] - vz[iz, ix - m])
                sxz[iz, ix] += dt * mu_xz[iz, ix] * (
                    dvx_dz / dz + dvz_dx / dx
                )

        # Isotropic explosive source: Mxx = Mzz; it primarily excites P waves.
        # sxx[iz_src, ix_src] += wavelet[it]
        szz[iz_src, ix_src] += wavelet[it]

        if snapshot_id < ns and it == snapshot_steps[snapshot_id]:
            snapshots_sxx[snapshot_id, :, :] = sxx
            snapshots_szz[snapshot_id, :, :] = szz
            snapshots_vx[snapshot_id, :, :] = vx
            snapshots_vz[snapshot_id, :, :] = vz
            snapshots_sxz[snapshot_id, :, :] = sxz
            snapshot_id += 1

    return snapshots_sxx, snapshots_szz, snapshots_sxz, snapshots_vx, snapshots_vz


def elastic_solver_dirichlet(
    wavelet: np.ndarray,
    coeff: np.ndarray,
    ix_src: int,
    iz_src: int,
    dx: float,
    dz: float,
    dt: float,
    rho: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    snapshot_steps: np.ndarray,
):
    """Solve the 2-D isotropic elastic equation using a staggered grid.

    ``rho``, ``lam``, and ``mu`` are ``(nz, nx)`` arrays at normal-stress
    nodes.  The returned tuple is ``(sxx, szz, sxz, vx, vz)``; its staggered
    array shapes are retained instead of silently discarding boundary faces.

    The source wavelet is added to both normal stresses.  Its samples therefore
    have units of stress increment (Pa), following the convention in the
    pressure-source code in the question.
    """
    wavelet = np.ascontiguousarray(wavelet, dtype=np.float64)
    coeff = np.ascontiguousarray(coeff, dtype=np.float64)
    rho = np.ascontiguousarray(rho, dtype=np.float64)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    mu = np.ascontiguousarray(mu, dtype=np.float64)
    snapshot_steps = np.ascontiguousarray(snapshot_steps, dtype=np.int64)

    if rho.ndim != 2 or lam.shape != rho.shape or mu.shape != rho.shape:
        raise ValueError("rho, lam, and mu must be 2-D arrays with equal shapes")
    if wavelet.ndim != 1 or coeff.ndim != 1 or len(coeff) == 0:
        raise ValueError("wavelet and nonempty coeff must be 1-D arrays")
    if np.any(rho <= 0.0) or np.any(mu <= 0.0) or np.any(lam + 2.0 * mu <= 0.0):
        raise ValueError("rho, mu, and lambda + 2*mu must be positive")
    if dx <= 0.0 or dz <= 0.0 or dt <= 0.0:
        raise ValueError("dx, dz, and dt must be positive")

    nz, nx = rho.shape
    n = len(coeff)
    if nz <= 2 * n or nx <= 2 * n:
        raise ValueError("the model must be larger than twice the half-stencil width")
    if ix_src < n or ix_src >= nx - n or iz_src < n or iz_src >= nz - n:
        raise ValueError("source is too close to the Dirichlet boundary")
    if snapshot_steps.ndim != 1:
        raise ValueError("snapshot_steps must be a 1-D integer array")
    if len(snapshot_steps) > 0:
        if snapshot_steps[0] < 0 or snapshot_steps[-1] >= len(wavelet):
            raise ValueError("snapshot_steps must lie between 0 and nt-1")
        if np.any(snapshot_steps[1:] <= snapshot_steps[:-1]):
            raise ValueError("snapshot_steps must be strictly increasing")

    vp_max = np.sqrt(np.max((lam + 2.0 * mu) / rho))
    courant = vp_max * dt * np.sqrt(1.0 / dx**2 + 1.0 / dz**2)
    if courant >= 0.6:
        raise ValueError(
            f"time step is probably unstable: Courant number = {courant:.3f}; "
            "reduce dt so it is below about 0.6"
        )

    return _elastic_solver_dirichlet_core(
        wavelet,
        coeff,
        ix_src,
        iz_src,
        dx,
        dz,
        dt,
        rho,
        lam,
        mu,
        snapshot_steps,
    )


if __name__ == "__main__":
    # Small homogeneous example. The first call includes Numba compilation time.
    if not NUMBA_AVAILABLE:
        raise SystemExit(
            "Numba is required for the full example. Install it with "
            "`uv pip install numba`, then run this file again."
        )
    nz, nx = 151, 151
    dx = dz = 10.0
    dt = 5.0e-4
    nt = 601

    rho = np.full((nz, nx), 2200.0)
    vp = np.full((nz, nx), 3000.0)
    vs = np.full((nz, nx), 1700.0)
    lam, mu = elastic_parameters(vp, vs, rho)

    coeff = staggered_fd_coefficients(order=8)
    wavelet = ricker_wavelet(f0=20.0, dt=dt, nt=nt, amplitude=1.0e7)
    snapshot_steps = np.array([200, 400, 600], dtype=np.int64)

    fields = elastic_solver_dirichlet(
        wavelet,
        coeff,
        ix_src=nx // 2,
        iz_src=nz // 2,
        dx=dx,
        dz=dz,
        dt=dt,
        rho=rho,
        lam=lam,
        mu=mu,
        snapshot_steps=snapshot_steps,
    )
    snapshots_sxx, snapshots_szz, snapshots_sxz, snapshots_vx, snapshots_vz = fields
    print("sxx snapshots:", snapshots_sxx.shape)
    print("szz snapshots:", snapshots_szz.shape)
    print("sxz snapshots:", snapshots_sxz.shape)
    print("vx snapshots: ", snapshots_vx.shape)
    print("vz snapshots: ", snapshots_vz.shape)
    print("maximum |sxx|:", np.max(np.abs(snapshots_sxx)))

    import matplotlib.pyplot as plt

    wavefield = snapshots_sxx[-1]
    vmax = np.max(np.abs(wavefield))
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        wavefield,extent=[
            0,
            (nx - 1) * dx,
            (nz - 1) * dz,
            0,
            ],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            )
    plt.scatter(
        (nx // 2) * dx,
        (nz // 2) * dz,
        marker="*",
        s=120,
        color="yellow",
        edgecolor="black",
        )
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(r"$\sigma_{xx}$")
    plt.colorbar(im, label=r"$\sigma_{xx}$ (Pa)")
    plt.show()

    wavefield = snapshots_szz[-1]
    vmax = np.max(np.abs(wavefield))
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        wavefield,extent=[
            0,
            (nx - 1) * dx,
            (nz - 1) * dz,
            0,
            ],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            )
    plt.scatter(
        (nx // 2) * dx,
        (nz // 2) * dz,
        marker="*",
        s=120,
        color="yellow",
        edgecolor="black",
        )
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(r"$\sigma_{zz}$")
    plt.colorbar(im, label=r"$\sigma_{zz}$ (Pa)")
    plt.show()    

    wavefield = snapshots_sxz[-1]
    vmax = np.max(np.abs(wavefield))
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        wavefield,extent=[
            0,
            (nx - 1) * dx,
            (nz - 1) * dz,
            0,
            ],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            )
    plt.scatter(
        (nx // 2) * dx,
        (nz // 2) * dz,
        marker="*",
        s=120,
        color="yellow",
        edgecolor="black",
        )
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(r"$\sigma_{xz}$")
    plt.colorbar(im, label=r"$\sigma_{xz}$ (Pa)")
    plt.show()

    wavefield = snapshots_vx[-1]
    vmax = np.max(np.abs(wavefield))
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        wavefield,extent=[
            0,
            (nx - 1) * dx,
            (nz - 1) * dz,
            0,
            ],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            )
    plt.scatter(
        (nx // 2) * dx,
        (nz // 2) * dz,
        marker="*",
        s=120,
        color="yellow",
        edgecolor="black",
        )
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(r"$v_{x}$")
    plt.colorbar(im, label=r"$v_{x}$ (Pa)")
    plt.show()

    wavefield = snapshots_vz[-1]
    vmax = np.max(np.abs(wavefield))
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        wavefield,extent=[
            0,
            (nx - 1) * dx,
            (nz - 1) * dz,
            0,
            ],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            )
    plt.scatter(
        (nx // 2) * dx,
        (nz // 2) * dz,
        marker="*",
        s=120,
        color="yellow",
        edgecolor="black",
        )
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(r"$v_{z}$")
    plt.colorbar(im, label=r"$v_{z}$ (Pa)")
    plt.show()