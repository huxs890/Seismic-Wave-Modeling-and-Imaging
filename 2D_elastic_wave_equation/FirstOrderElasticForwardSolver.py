import numpy as np
from dataclasses import dataclass
from staggered_fd_coefficients import staggered_fd_coefficients
from VTI_PML_boundary import VTI_PML_boundary
from _elastic_solver_dirichlet import (
     _elastic_vti_solver_dirichlet
)
from _elastic_solver_pml import (
    _elastic_vti_solver_split_PML
)
# ============================================================
# Output
# ============================================================
@dataclass
class ElasticResult:
    snapshots_vx: np.ndarray
    snapshots_vz: np.ndarray
    seismogram_vx: np.ndarray
    seismogram_vz: np.ndarray

# ============================================================
# First-order VTI elastic forward solver
# ============================================================
class FirstOrderElasticForwardSolver:
    def __init__(
        self,
        rho,
        c11,
        c13,
        c33,
        c55,
        dx,
        dz,
        dt,
        order=8,
    ):
        # ====================================================
        # 1. Model
        # ====================================================
        self.rho = np.asarray(rho,dtype=np.float32)
        self.c11 = np.asarray(c11,dtype=np.float32)
        self.c13 = np.asarray(c13,dtype=np.float32)
        self.c33 = np.asarray(c33,dtype=np.float32)
        self.c55 = np.asarray(c55,dtype=np.float32)
        self.dx = np.float32(dx)
        self.dz = np.float32(dz)
        self.dt = np.float32(dt)
        self.order = np.int32(order)
        # ----------------------------------------------------
        # Model shape
        # ----------------------------------------------------
        if self.rho.ndim != 2:
            raise ValueError("Model parameters must be 2-D arrays.")
        if (
            self.c11.shape != self.rho.shape
            or self.c13.shape != self.rho.shape
            or self.c33.shape != self.rho.shape
            or self.c55.shape != self.rho.shape
        ):
            raise ValueError("rho and Cij must have the same shape.")
        self.nz, self.nx = self.rho.shape
        # ====================================================
        # 2. Staggered-grid FD coefficients
        # ====================================================
        self.coeff = np.asarray(staggered_fd_coefficients(int(self.order)),dtype=np.float32)
        # ====================================================
        # 3. Source
        # ====================================================
        self.wavelet = None
        self.ix_src = None
        self.iz_src = None
        # ====================================================
        # 4. Receivers
        # ====================================================
        self.ix_receivers = None
        self.iz_receivers = None
        # ====================================================
        # 5. Boundary
        # ====================================================
        self.boundary_type = None
        self.n_abs = None
        self.rho_ext = None
        self.c11_ext = None
        self.c13_ext = None
        self.c33_ext = None
        self.c55_ext = None
        self.sigma_x = None
        self.sigma_z = None

    # ========================================================
    # Set source
    # ========================================================
    def set_source(
        self,
        wavelet,
        ix,
        iz,
    ):
        wavelet = np.asarray(wavelet,dtype=np.float32)
        ix = np.int32(ix)
        iz = np.int32(iz)
        # ----------------------------------------------------
        # Wavelet
        # ----------------------------------------------------
        if wavelet.ndim != 1:
            raise ValueError("wavelet must be a 1-D array.")
        if len(wavelet) < 2:
            raise ValueError("wavelet must contain at least two samples.")
        # ----------------------------------------------------
        # Source position
        # ----------------------------------------------------
        if ix < 0 or ix >= self.nx:
            raise ValueError("Source x-position is outside the model.")
        if iz < 0 or iz >= self.nz:
            raise ValueError("Source z-position is outside the model.")
        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------
        self.wavelet = wavelet
        self.ix_src = ix
        self.iz_src = iz

    # ========================================================
    # Set receivers
    # ========================================================
    def set_receivers(
        self,
        ix,
        iz,
    ):
        ix = np.atleast_1d(np.asarray(ix,dtype=np.int32))
        iz = np.atleast_1d(np.asarray(iz,dtype=np.int32))
        # ----------------------------------------------------
        # Basic checks
        # ----------------------------------------------------
        if len(ix) != len(iz):
            raise ValueError("ix and iz must have the same length.")
        if ix.ndim != 1 or iz.ndim != 1:raise ValueError("Receiver indices must be 1-D arrays.")
        if ix.size == 0:
            raise ValueError("At least one receiver is required.")
        # ----------------------------------------------------
        # Receiver positions
        # ----------------------------------------------------
        if np.any(ix < 0) or np.any(ix >= self.nx):
            raise ValueError("Receiver x-index is outside the model.")
        if np.any(iz < 0) or np.any(iz >= self.nz):
            raise ValueError("Receiver z-index is outside the model.")
        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------
        self.ix_receivers = ix
        self.iz_receivers = iz
    
    # ========================================================
    # Set boundary condition
    # ========================================================
    def set_boundary(
        self,
        boundary_type="pml",
        n_abs=40,
        reflection=1e-6,
        power=2,
    ):
        boundary_type = str(boundary_type).lower()
        # ====================================================
        # Dirichlet
        # ====================================================
        if boundary_type == "dirichlet":
            self.boundary_type = "dirichlet"
            self.n_abs = None
            self.rho_ext = None
            self.c11_ext = None
            self.c13_ext = None
            self.c33_ext = None
            self.c55_ext = None
            self.sigma_x = None
            self.sigma_z = None
        # ====================================================
        # PML
        # ====================================================
        elif boundary_type == "pml":
            n_abs = np.int32(n_abs)
            if n_abs < len(self.coeff):
                raise ValueError("n_abs must be >= half stencil width.")
            (
                rho_ext,
                c11_ext,
                c13_ext,
                c33_ext,
                c55_ext,
                sigma_x,
                sigma_z,
            ) = VTI_PML_boundary(
                self.rho,
                self.c11,
                self.c13,
                self.c33,
                self.c55,
                self.dx,
                self.dz,
                n_abs=n_abs,
                reflection=reflection,
                power=power,
            )

            # ------------------------------------------------
            # Save PML model
            # ------------------------------------------------
            self.boundary_type = "pml"
            self.n_abs = np.int32(n_abs)
            self.rho_ext = np.asarray(rho_ext,dtype=np.float32)
            self.c11_ext = np.asarray(c11_ext,dtype=np.float32)
            self.c13_ext = np.asarray(c13_ext,dtype=np.float32)
            self.c33_ext = np.asarray(c33_ext,dtype=np.float32)
            self.c55_ext = np.asarray(c55_ext,dtype=np.float32)
            self.sigma_x = np.asarray(sigma_x,dtype=np.float32)
            self.sigma_z = np.asarray(sigma_z,dtype=np.float32)
        else:
            raise ValueError(
                "boundary_type must be 'dirichlet' or 'pml'."
                )

    # ========================================================
    # Forward modeling
    # ========================================================
    def forward(
        self,
        snapshot_times=None,
    ):
        # ====================================================
        # 1. Check source / receivers / boundary
        # ====================================================
        if self.wavelet is None:
            raise RuntimeError(
                "Source has not been defined."
            )
        if self.ix_receivers is None:
            raise RuntimeError(
                "Receivers have not been defined."
            )
        if self.boundary_type is None:
            raise RuntimeError(
                "Boundary condition has not been defined."
            )
        nt = len(self.wavelet)

        # ====================================================
        # 2. Snapshot times -> steps
        # ====================================================
        if snapshot_times is None:
            snapshot_steps = np.empty(0,dtype=np.int32)
        else:
            snapshot_times = np.atleast_1d(np.asarray(snapshot_times,dtype=np.float32))
            snapshot_steps = np.rint(snapshot_times/self.dt).astype(np.int32)
            if np.any(snapshot_steps < 0) or np.any(snapshot_steps >= nt):
                raise ValueError(
                    "snapshot_times are outside "
                    "the simulation time range."
                )
            if len(snapshot_steps) > 1:
                if np.any(np.diff(snapshot_steps) <= 0):
                    raise ValueError(
                        "snapshot_times must be "
                        "strictly increasing."
                    )
                
        # ====================================================
        # 3. Dirichlet solver
        # ====================================================
        if self.boundary_type == "dirichlet":
            (
                seismogram_vx,
                seismogram_vz,
                snapshots_vx,
                snapshots_vz,
            ) = _elastic_vti_solver_dirichlet(
                self.rho,
                self.c11,
                self.c13,
                self.c33,
                self.c55,
                self.dx,
                self.dz,
                self.dt,
                self.coeff,
                self.ix_src,
                self.iz_src,
                self.wavelet,
                self.ix_receivers,
                self.iz_receivers,
                snapshot_steps,
            )

        # ====================================================
        # 4. Split-field PML solver
        # ====================================================
        elif self.boundary_type == "pml":
            (
                seismogram_vx,
                seismogram_vz,
                snapshots_vx,
                snapshots_vz,
            ) = _elastic_vti_solver_split_PML(
                self.rho_ext,
                self.c11_ext,
                self.c13_ext,
                self.c33_ext,
                self.c55_ext,
                self.sigma_x,
                self.sigma_z,
                self.n_abs,
                self.dx,
                self.dz,
                self.dt,
                self.coeff,
                self.ix_src,
                self.iz_src,
                self.wavelet,
                self.ix_receivers,
                self.iz_receivers,
                snapshot_steps,
            )
        else:
            raise RuntimeError(
                f"Unknown boundary type: "
                f"{self.boundary_type}"
            )
        # ====================================================
        # 5. Output
        # ====================================================
        return ElasticResult(
            snapshots_vx=snapshots_vx,
            snapshots_vz=snapshots_vz,
            seismogram_vx=seismogram_vx,
            seismogram_vz=seismogram_vz,
        )