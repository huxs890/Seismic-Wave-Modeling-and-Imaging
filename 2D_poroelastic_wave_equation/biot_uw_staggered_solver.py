"""2-D dynamic Biot poroelastic wave solver in the u-w formulation.

Plane-strain, isotropic, low-frequency Biot equations are discretized on a
Virieux staggered grid.  The solid displacement is u and

    w = phi * (U - u)

is the relative fluid displacement per unit bulk area.  Consequently q = w_t
has units of Darcy flux.  A Crank-Nicolson treatment of the Darcy drag makes
the viscous part stable even when permeability is small.

This file is self-contained: running it produces pore-pressure snapshots and a
centre-line profile on which the fast- and slow-P wavefronts are marked.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np


def staggered_fd_coefficients(order: int) -> np.ndarray:
    """Return 2N-order coefficients for a staggered first derivative."""
    if order <= 0 or order % 2:
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


def ricker_wavelet(f0: float, dt: float, nt: int, amplitude: float) -> np.ndarray:
    """Ricker wavelet with its maximum at t0 = 1.5/f0."""
    t = np.arange(nt, dtype=np.float64) * dt
    t0 = 1.5 / f0
    a = (np.pi * f0 * (t - t0)) ** 2
    return amplitude * (1.0 - 2.0 * a) * np.exp(-a)


def biot_material(
    shape: tuple[int, int],
    grain_bulk: float,
    drained_bulk: float,
    shear: float,
    porosity: float,
    fluid_bulk: float,
    grain_density: float,
    fluid_density: float,
    tortuosity: float,
    permeability: float,
    viscosity: float,
) -> dict[str, np.ndarray]:
    """Build homogeneous low-frequency Biot parameter arrays.

    ``rho`` is the saturated bulk density, ``m = tortuosity*rho_f/phi``, and
    ``drag = viscosity/permeability``.  The drained Lame parameter is based on
    the three-dimensional bulk modulus; the 2-D computation is plane strain.
    """
    if not (0.0 < porosity < 1.0):
        raise ValueError("porosity must lie between zero and one")
    positive = (
        grain_bulk,
        drained_bulk,
        shear,
        fluid_bulk,
        grain_density,
        fluid_density,
        tortuosity,
        permeability,
        viscosity,
    )
    if any(value <= 0.0 for value in positive):
        raise ValueError("all elastic, density, flow, and tortuosity values must be positive")
    if drained_bulk >= grain_bulk:
        raise ValueError("drained_bulk must be smaller than grain_bulk")

    alpha_value = 1.0 - drained_bulk / grain_bulk
    storage = (alpha_value - porosity) / grain_bulk + porosity / fluid_bulk
    if storage <= 0.0:
        raise ValueError("the selected parameters give a non-positive Biot modulus")
    biot_modulus_value = 1.0 / storage
    lambda_d_value = drained_bulk - 2.0 * shear / 3.0
    if lambda_d_value <= 0.0:
        raise ValueError("drained_bulk - 2*shear/3 must be positive in this example")

    bulk_density = (1.0 - porosity) * grain_density + porosity * fluid_density
    inertial_mass = tortuosity * fluid_density / porosity

    def field(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float64)

    return {
        "rho": field(bulk_density),
        "rho_f": field(fluid_density),
        "lambda_d": field(lambda_d_value),
        "mu": field(shear),
        "alpha": field(alpha_value),
        "M": field(biot_modulus_value),
        "m": field(inertial_mass),
        "drag": field(viscosity / permeability),
    }


def compressional_wave_speeds(material: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return maximum local inviscid fast- and slow-P phase speeds."""
    rho = material["rho"]
    rho_f = material["rho_f"]
    lam_d = material["lambda_d"]
    mu = material["mu"]
    alpha = material["alpha"]
    biot_m = material["M"]
    inertial_m = material["m"]

    q_coupling = alpha * biot_m
    h_modulus = lam_d + 2.0 * mu + alpha**2 * biot_m
    determinant = rho * inertial_m - rho_f**2
    middle = (
        h_modulus * inertial_m
        + biot_m * rho
        - 2.0 * q_coupling * rho_f
    )
    constant = h_modulus * biot_m - q_coupling**2
    discriminant = np.maximum(middle**2 - 4.0 * determinant * constant, 0.0)
    c_fast_2 = (middle + np.sqrt(discriminant)) / (2.0 * determinant)
    c_slow_2 = (middle - np.sqrt(discriminant)) / (2.0 * determinant)
    return float(np.sqrt(np.max(c_fast_2))), float(np.sqrt(np.max(c_slow_2)))


def _x_face(field: np.ndarray) -> np.ndarray:
    """Arithmetic average a cell-centred field onto x faces."""
    nz, nx = field.shape
    result = np.empty((nz, nx + 1), dtype=np.float64)
    result[:, 1:nx] = 0.5 * (field[:, :-1] + field[:, 1:])
    result[:, 0] = field[:, 0]
    result[:, nx] = field[:, -1]
    return result


def _z_face(field: np.ndarray) -> np.ndarray:
    """Arithmetic average a cell-centred field onto z faces."""
    nz, nx = field.shape
    result = np.empty((nz + 1, nx), dtype=np.float64)
    result[1:nz, :] = 0.5 * (field[:-1, :] + field[1:, :])
    result[0, :] = field[0, :]
    result[nz, :] = field[-1, :]
    return result


def _corner_harmonic(field: np.ndarray) -> np.ndarray:
    """Four-cell harmonic average onto the shear-stress corners."""
    nz, nx = field.shape
    result = np.zeros((nz + 1, nx + 1), dtype=np.float64)
    result[1:nz, 1:nx] = 4.0 / (
        1.0 / field[:-1, :-1]
        + 1.0 / field[:-1, 1:]
        + 1.0 / field[1:, :-1]
        + 1.0 / field[1:, 1:]
    )
    return result


def _validate_inputs(
    wavelet,
    coeff,
    ix_src,
    iz_src,
    dx,
    dz,
    dt,
    material,
    snapshot_steps,
):
    required = ("rho", "rho_f", "lambda_d", "mu", "alpha", "M", "m", "drag")
    if any(name not in material for name in required):
        raise ValueError(f"material must contain {required}")
    shape = np.asarray(material["rho"]).shape
    if len(shape) != 2 or any(np.asarray(material[name]).shape != shape for name in required):
        raise ValueError("all material entries must be equal-sized 2-D arrays")
    if any(np.any(np.asarray(material[name]) <= 0.0) for name in ("rho", "rho_f", "mu", "M", "m", "drag")):
        raise ValueError("mass, modulus, and drag parameters must be positive")
    if np.any(np.asarray(material["alpha"]) < 0.0):
        raise ValueError("Biot alpha must be non-negative")
    if dx <= 0.0 or dz <= 0.0 or dt <= 0.0:
        raise ValueError("dx, dz, and dt must be positive")
    if np.ndim(wavelet) != 1 or np.ndim(coeff) != 1 or len(coeff) == 0:
        raise ValueError("wavelet and nonempty coeff must be one-dimensional")
    n = len(coeff)
    nz, nx = shape
    if nz <= 2 * n or nx <= 2 * n:
        raise ValueError("model is too small for the chosen stencil")
    if not (n <= ix_src < nx - n and n <= iz_src < nz - n):
        raise ValueError("source is too close to the frozen boundary")
    steps = np.asarray(snapshot_steps)
    if steps.ndim != 1 or (len(steps) and (steps[0] < 0 or steps[-1] >= len(wavelet))):
        raise ValueError("snapshot_steps must be a valid one-dimensional index array")
    if len(steps) > 1 and np.any(steps[1:] <= steps[:-1]):
        raise ValueError("snapshot_steps must be strictly increasing")
    mass_det = np.asarray(material["rho"]) * np.asarray(material["m"]) - np.asarray(material["rho_f"])**2
    if np.any(mass_det <= 0.0):
        raise ValueError("rho*m-rho_f**2 must be positive")
    c_fast, _ = compressional_wave_speeds(material)
    courant = c_fast * dt * np.sqrt(dx**-2 + dz**-2)
    if courant >= 0.55:
        raise ValueError(
            f"Courant number {courant:.3f} is too large; reduce dt below "
            f"{0.5 / (c_fast * np.sqrt(dx**-2 + dz**-2)):.3e} s"
        )


def biot_uw_solver_dirichlet(
    wavelet: np.ndarray,
    coeff: np.ndarray,
    ix_src: int,
    iz_src: int,
    dx: float,
    dz: float,
    dt: float,
    material: dict[str, np.ndarray],
    snapshot_steps: np.ndarray,
    source_type: str = "pore",
    progress_interval: int = 100,
) -> dict[str, np.ndarray]:
    """Solve the dynamic Biot equations with a staggered-grid u-w scheme.

    Parameters
    ----------
    source_type : {"pore", "bulk"}
        ``pore`` adds a pressure increment and strongly illuminates the slow-P
        mode. ``bulk`` adds equal normal-stress increments and mainly excites
        the fast-P mode.

    Returns
    -------
    dict
        Snapshots of p, sxx, szz, sxz, ux, uz, wx, and wz.  Face and corner
        fields retain their true staggered array sizes.
    """
    wavelet = np.asarray(wavelet, dtype=np.float64)
    coeff = np.asarray(coeff, dtype=np.float64)
    steps = np.asarray(snapshot_steps, dtype=np.int64)
    material = {name: np.asarray(value, dtype=np.float64) for name, value in material.items()}
    _validate_inputs(wavelet, coeff, ix_src, iz_src, dx, dz, dt, material, steps)
    if source_type not in ("pore", "bulk"):
        raise ValueError("source_type must be 'pore' or 'bulk'")

    rho = material["rho"]
    rho_f = material["rho_f"]
    lam_d = material["lambda_d"]
    mu = material["mu"]
    alpha = material["alpha"]
    biot_m = material["M"]
    inertial_m = material["m"]
    drag = material["drag"]
    nz, nx = rho.shape
    n = len(coeff)
    ns = len(steps)

    # Normal stress and pore pressure at cell centres.
    sxx = np.zeros((nz, nx))
    szz = np.zeros((nz, nx))
    pressure = np.zeros((nz, nx))
    # Solid velocity v and Darcy flux q=w_t at faces.
    vx = np.zeros((nz, nx + 1))
    vz = np.zeros((nz + 1, nx))
    qx = np.zeros((nz, nx + 1))
    qz = np.zeros((nz + 1, nx))
    # Displacements use the same faces as their velocities.
    ux = np.zeros_like(vx)
    uz = np.zeros_like(vz)
    wx = np.zeros_like(qx)
    wz = np.zeros_like(qz)
    sxz = np.zeros((nz + 1, nx + 1))

    mu_corner = _corner_harmonic(mu)
    face_x = {name: _x_face(material[name]) for name in ("rho", "rho_f", "m", "drag")}
    face_z = {name: _z_face(material[name]) for name in ("rho", "rho_f", "m", "drag")}

    snapshots = {
        "p": np.zeros((ns, nz, nx)),
        "sxx": np.zeros((ns, nz, nx)),
        "szz": np.zeros((ns, nz, nx)),
        "sxz": np.zeros((ns, nz + 1, nx + 1)),
        "ux": np.zeros((ns, nz, nx + 1)),
        "uz": np.zeros((ns, nz + 1, nx)),
        "wx": np.zeros((ns, nz, nx + 1)),
        "wz": np.zeros((ns, nz + 1, nx)),
    }
    snapshot_id = 0

    zc = slice(n, nz - n)
    xc = slice(n, nx - n)
    zvx = slice(n, nz - n)
    xvx = slice(n, nx - n + 1)
    zvz = slice(n, nz - n + 1)
    xvz = slice(n, nx - n)
    zcorner = slice(n, nz - n + 1)
    xcorner = slice(n, nx - n + 1)

    for it in range(len(wavelet)):

        # Progress
        if (it + 1) % 100 == 0 or it == nt - 1:
            print("Time step:",it + 1,"/",nt,"  Time:",(it + 1) * dt,"s")

        # Stress divergence and pressure gradient on x faces.
        force_x = np.zeros((nz - 2 * n, nx - 2 * n + 1))
        gradp_x = np.zeros_like(force_x)
        for m_index, c in enumerate(coeff, start=1):
            force_x += c * (
                (sxx[zc, n + m_index - 1 : nx - n + m_index]
                 - sxx[zc, n - m_index : nx - n - m_index + 1]) / dx
                + (sxz[n + m_index : nz - n + m_index, xvx]
                   - sxz[n - m_index + 1 : nz - n - m_index + 1, xvx]) / dz
            )
            gradp_x += c * (
                pressure[zc, n + m_index - 1 : nx - n + m_index]
                - pressure[zc, n - m_index : nx - n - m_index + 1]
            ) / dx

        # Crank-Nicolson Darcy drag; solve the local 2x2 mass system exactly.
        rho_x = face_x["rho"][zvx, xvx]
        rhof_x = face_x["rho_f"][zvx, xvx]
        mass_x = face_x["m"][zvx, xvx]
        drag_x = face_x["drag"][zvx, xvx]
        old_vx = vx[zvx, xvx]
        old_qx = qx[zvx, xvx]
        rhs1 = rho_x * old_vx + rhof_x * old_qx + dt * force_x
        rhs2 = (
            rhof_x * old_vx
            + (mass_x - 0.5 * dt * drag_x) * old_qx
            - dt * gradp_x
        )
        mass_drag_x = mass_x + 0.5 * dt * drag_x
        determinant_x = rho_x * mass_drag_x - rhof_x**2
        vx[zvx, xvx] = (mass_drag_x * rhs1 - rhof_x * rhs2) / determinant_x
        qx[zvx, xvx] = (-rhof_x * rhs1 + rho_x * rhs2) / determinant_x

        # Corresponding z-face update.
        force_z = np.zeros((nz - 2 * n + 1, nx - 2 * n))
        gradp_z = np.zeros_like(force_z)
        for m_index, c in enumerate(coeff, start=1):
            force_z += c * (
                (sxz[zvz, n + m_index : nx - n + m_index]
                 - sxz[zvz, n - m_index + 1 : nx - n - m_index + 1]) / dx
                + (szz[n + m_index - 1 : nz - n + m_index, xvz]
                   - szz[n - m_index : nz - n - m_index + 1, xvz]) / dz
            )
            gradp_z += c * (
                pressure[n + m_index - 1 : nz - n + m_index, xvz]
                - pressure[n - m_index : nz - n - m_index + 1, xvz]
            ) / dz

        rho_z = face_z["rho"][zvz, xvz]
        rhof_z = face_z["rho_f"][zvz, xvz]
        mass_z = face_z["m"][zvz, xvz]
        drag_z = face_z["drag"][zvz, xvz]
        old_vz = vz[zvz, xvz]
        old_qz = qz[zvz, xvz]
        rhs1 = rho_z * old_vz + rhof_z * old_qz + dt * force_z
        rhs2 = (
            rhof_z * old_vz
            + (mass_z - 0.5 * dt * drag_z) * old_qz
            - dt * gradp_z
        )
        mass_drag_z = mass_z + 0.5 * dt * drag_z
        determinant_z = rho_z * mass_drag_z - rhof_z**2
        vz[zvz, xvz] = (mass_drag_z * rhs1 - rhof_z * rhs2) / determinant_z
        qz[zvz, xvz] = (-rhof_z * rhs1 + rho_z * rhs2) / determinant_z

        # v and q are at n+1/2, so these are centred displacement updates.
        ux[zvx, xvx] += dt * vx[zvx, xvx]
        wx[zvx, xvx] += dt * qx[zvx, xvx]
        uz[zvz, xvz] += dt * vz[zvz, xvz]
        wz[zvz, xvz] += dt * qz[zvz, xvz]

        div_vx = np.zeros((nz - 2 * n, nx - 2 * n))
        div_vz = np.zeros_like(div_vx)
        div_qx = np.zeros_like(div_vx)
        div_qz = np.zeros_like(div_vx)
        for m_index, c in enumerate(coeff, start=1):
            div_vx += c * (
                vx[zc, n + m_index : nx - n + m_index]
                - vx[zc, n - m_index + 1 : nx - n - m_index + 1]
            ) / dx
            div_qx += c * (
                qx[zc, n + m_index : nx - n + m_index]
                - qx[zc, n - m_index + 1 : nx - n - m_index + 1]
            ) / dx
            div_vz += c * (
                vz[n + m_index : nz - n + m_index, xc]
                - vz[n - m_index + 1 : nz - n - m_index + 1, xc]
            ) / dz
            div_qz += c * (
                qz[n + m_index : nz - n + m_index, xc]
                - qz[n - m_index + 1 : nz - n - m_index + 1, xc]
            ) / dz

        div_v = div_vx + div_vz
        div_q = div_qx + div_qz
        lam_u = lam_d[zc, xc] + alpha[zc, xc] ** 2 * biot_m[zc, xc]
        coupling = alpha[zc, xc] * biot_m[zc, xc]
        sxx[zc, xc] += dt * (
            (lam_u + 2.0 * mu[zc, xc]) * div_vx
            + lam_u * div_vz
            + coupling * div_q
        )
        szz[zc, xc] += dt * (
            lam_u * div_vx
            + (lam_u + 2.0 * mu[zc, xc]) * div_vz
            + coupling * div_q
        )
        pressure[zc, xc] -= dt * biot_m[zc, xc] * (
            alpha[zc, xc] * div_v + div_q
        )

        dvx_dz = np.zeros((nz - 2 * n + 1, nx - 2 * n + 1))
        dvz_dx = np.zeros_like(dvx_dz)
        for m_index, c in enumerate(coeff, start=1):
            dvx_dz += c * (
                vx[n + m_index - 1 : nz - n + m_index, xcorner]
                - vx[n - m_index : nz - n - m_index + 1, xcorner]
            ) / dz
            dvz_dx += c * (
                vz[zcorner, n + m_index - 1 : nx - n + m_index]
                - vz[zcorner, n - m_index : nx - n - m_index + 1]
            ) / dx
        sxz[zcorner, xcorner] += dt * mu_corner[zcorner, xcorner] * (dvx_dz + dvz_dx)

        if source_type == "pore":
            pressure[iz_src, ix_src] += wavelet[it]
        else:
            sxx[iz_src, ix_src] += wavelet[it]
            szz[iz_src, ix_src] += wavelet[it]

        if snapshot_id < ns and it == steps[snapshot_id]:
            snapshots["p"][snapshot_id] = pressure
            snapshots["sxx"][snapshot_id] = sxx
            snapshots["szz"][snapshot_id] = szz
            snapshots["sxz"][snapshot_id] = sxz
            snapshots["ux"][snapshot_id] = ux
            snapshots["uz"][snapshot_id] = uz
            snapshots["wx"][snapshot_id] = wx
            snapshots["wz"][snapshot_id] = wz
            snapshot_id += 1

        if progress_interval > 0 and ((it + 1) % progress_interval == 0 or it + 1 == len(wavelet)):
            print(f"Time step {it + 1}/{len(wavelet)}, time = {(it + 1) * dt:.5f} s")

    return snapshots


def plot_pressure_snapshots(
    snapshots: dict[str, np.ndarray],
    snapshot_steps: np.ndarray,
    dt: float,
    dx: float,
    dz: float,
    ix_src: int,
    iz_src: int,
    c_fast: float,
    c_slow: float,
    source_peak_time: float,
    output: str | None = None,
):
    """Plot pressure snapshots and predicted inviscid P-wavefront circles."""
    pressure = snapshots["p"]
    nz, nx = pressure.shape[1:]
    ns = len(snapshot_steps)
    fig, axes = plt.subplots(1, ns, figsize=(5.2 * ns, 4.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    extent = [0.0, (nx - 1) * dx, (nz - 1) * dz, 0.0]
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    source_x = ix_src * dx
    source_z = iz_src * dz

    for index, (axis, step) in enumerate(zip(axes, snapshot_steps)):
        field = pressure[index]
        clip = np.percentile(np.abs(field), 99.7)
        if clip == 0.0:
            clip = 1.0
        image = axis.imshow(
            field,
            extent=extent,
            cmap="seismic",
            vmin=-clip/1e1,
            vmax=clip/1e1,
            interpolation="bilinear",
        )
        physical_time = (step + 1) * dt
        elapsed = max(physical_time - source_peak_time, 0.0)
        for speed, color, label in (
            (c_fast, "lime", "fast P"),
            (c_slow, "yellow", "slow P"),
        ):
            radius = speed * elapsed
            axis.plot(
                source_x + radius * np.cos(theta),
                source_z + radius * np.sin(theta),
                color=color,
                linewidth=1.2,
                linestyle="--",
                label=label,
            )
        axis.scatter(source_x, source_z, marker="*", s=60, color="black")
        axis.set_title(f"Pore pressure, t = {physical_time:.3f} s")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")
        axis.legend(loc="upper right", fontsize=8)
        fig.colorbar(image, ax=axis, label="p (Pa)", shrink=0.83)
    if output:
        fig.savefig(output, dpi=180)
    return fig


def plot_centre_profile(
    snapshots: dict[str, np.ndarray],
    final_step: int,
    dt: float,
    dx: float,
    ix_src: int,
    iz_src: int,
    c_fast: float,
    c_slow: float,
    source_peak_time: float,
    output: str | None = None,
):
    """Plot the final horizontal pressure profile to separate the two P modes."""
    profile = snapshots["p"][-1, iz_src]
    scale = np.max(np.abs(profile))
    if scale > 0.0:
        profile = profile / scale
    distance = (np.arange(profile.size) - ix_src) * dx
    physical_time = (final_step + 1) * dt
    elapsed = max(physical_time - source_peak_time, 0.0)
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.plot(distance, profile, color="black", linewidth=1.2, label="normalized p")
    for speed, color, label in (
        (c_fast, "green", "predicted fast P"),
        (c_slow, "darkorange", "predicted slow P"),
    ):
        radius = speed * elapsed
        axis.axvline(radius, color=color, linestyle="--", label=label)
        axis.axvline(-radius, color=color, linestyle="--")
    axis.set_xlabel("Horizontal distance from source (m)")
    axis.set_ylabel("Normalized pore pressure")
    axis.set_title(f"Centre-line profile at t = {physical_time:.3f} s")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    if output:
        fig.savefig(output, dpi=180)
    return fig


if __name__ == "__main__":
    nz = nx = 241
    dx = dz = 2.5
    dt = 2.5e-4
    nt = 401
    f0 = 40.0

    # High permeability makes f0 exceed the Biot transition frequency, so the
    # slow compressional mode is wave-like and visually distinguishable.
    phi = 0.25
    rho_f_value = 1000.0
    tortuosity = 2.5
    viscosity = 1.0e-3
    permeability = 1.0e-9
    material = biot_material(
        shape=(nz, nx),
        grain_bulk=36.0e9,
        drained_bulk=8.0e9,
        shear=6.0e9,
        porosity=phi,
        fluid_bulk=2.2e9,
        grain_density=2650.0,
        fluid_density=rho_f_value,
        tortuosity=tortuosity,
        permeability=permeability,
        viscosity=viscosity,
    )

    coeff = staggered_fd_coefficients(order=4)
    wavelet = ricker_wavelet(f0=f0, dt=dt, nt=nt, amplitude=2.0e6)
    snapshot_steps = np.array([220, 300, 400], dtype=np.int64)
    ix_src = nx // 2
    iz_src = nz // 2

    c_fast, c_slow = compressional_wave_speeds(material)
    biot_frequency = viscosity * phi / (
        2.0 * np.pi * permeability * rho_f_value * tortuosity
    )
    print(f"Inviscid fast-P speed: {c_fast:.1f} m/s")
    print(f"Inviscid slow-P speed: {c_slow:.1f} m/s")
    print(f"Approximate Biot transition frequency: {biot_frequency:.1f} Hz")
    print(f"Source dominant frequency: {f0:.1f} Hz")

    snapshots = biot_uw_solver_dirichlet(
        wavelet=wavelet,
        coeff=coeff,
        ix_src=ix_src,
        iz_src=iz_src,
        dx=dx,
        dz=dz,
        dt=dt,
        material=material,
        snapshot_steps=snapshot_steps,
        source_type="pore",
        progress_interval=100,
    )

    source_peak_time = 1.5 / f0
    plot_pressure_snapshots(
        snapshots,
        snapshot_steps,
        dt,
        dx,
        dz,
        ix_src,
        iz_src,
        c_fast,
        c_slow,
        source_peak_time,
        output="biot_uw_pressure_snapshots.png",
    )
    plot_centre_profile(
        snapshots,
        int(snapshot_steps[-1]),
        dt,
        dx,
        ix_src,
        iz_src,
        c_fast,
        c_slow,
        source_peak_time,
        output="biot_uw_centre_profile.png",
    )
    plt.show()
