import numpy as np
from staggered_fd_coefficients import staggered_fd_coefficients
from PML_boundary import PML_boundary
from dataclasses import dataclass
from _acoustic_solver_dirichlet import _acoustic_solver_dirichlet
from _acoustic_solver_split_PML import _acoustic_solver_split_PML

# ============================================================
# Output
# ============================================================
@dataclass
class AcousticResult:
    snapshots: np.ndarray
    seismogram: np.ndarray
# ============================================================
# Main routin
# ============================================================
class FirstOrderAcousticForwardSolver:
    def __init__(
        self,
        rho,
        bulk,
        dx,
        dz,
        dt,
        order=8,
    ):
        # ====================================================
        # Model
        # ====================================================
        self.rho = np.asarray(rho,dtype=np.float64)
        self.bulk = np.asarray(bulk,dtype=np.float64)
        self.dx = float(dx)
        self.dz = float(dz)
        self.dt = float(dt)
        self.order = int(order)
        self.nz, self.nx = self.rho.shape
        if self.rho.shape != self.bulk.shape:
            raise ValueError("rho and bulk must have the same shape.")
        if self.rho.ndim != 2:
            raise ValueError("rho and bulk must be 2-D arrays.")
        # ====================================================
        # FD coefficients
        # ====================================================
        self.coeff = np.asarray(staggered_fd_coefficients(self.order),dtype=np.float64)
        # ====================================================
        # Source
        # ====================================================
        self.wavelet = None
        self.ix_src = None
        self.iz_src = None
        # ====================================================
        # Receivers
        # ====================================================
        self.ix_receivers = None
        self.iz_receivers = None
        # ====================================================
        # Boundary
        # ====================================================
        self.boundary_type = None
        # PML parameters
        self.n_abs = None
        self.rho_ext = None
        self.bulk_ext = None
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
        wavelet = np.asarray(wavelet,dtype=np.float64)
        ix = int(ix)
        iz = int(iz)
        # ----------------------------------------------------
        # Wavelet
        # ----------------------------------------------------
        if wavelet.ndim != 1:raise ValueError("wavelet must be a 1-D array.")
        if len(wavelet) < 2:
            raise ValueError("wavelet must contain at least two time samples.")
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
        ix = np.atleast_1d(np.asarray(ix,dtype=np.int64))
        iz = np.atleast_1d(np.asarray(iz,dtype=np.int64))
        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------
        if len(ix) != len(iz):
            raise ValueError("ix and iz must have the same length.")
        if ix.ndim != 1 or iz.ndim != 1:
            raise ValueError("Receiver indices must be 1-D arrays.")
        if ix.size == 0:
            raise ValueError("At least one receiver is required.")
        # ----------------------------------------------------
        # Check positions
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
    # Set boundary
    # ========================================================
    def set_boundary(
        self,
        boundary_type="PML",
        n_abs=40,
        reflection=1e-6,
        power=2,
    ):
        boundary_type = str(boundary_type).lower()
        # ====================================================
        # Dirichlet boundary
        # ====================================================
        if boundary_type == "dirichlet":
            self.boundary_type = "dirichlet"
            # Clear PML parameters
            self.n_abs = None
            self.rho_ext = None
            self.bulk_ext = None
            self.sigma_x = None
            self.sigma_z = None
        # ====================================================
        # PML boundary
        # ====================================================
        elif boundary_type == "pml":
            n_abs = int(n_abs)
            # ------------------------------------------------
            # Generate extended models and damping fields
            # ------------------------------------------------
            rho_ext,bulk_ext,sigma_x,sigma_z = PML_boundary(
                self.rho,
                self.bulk,
                self.dx,
                self.dz,
                n_abs=n_abs,
                reflection=reflection,
                power=power,
            )
            # ------------------------------------------------
            # Save
            # ------------------------------------------------
            self.boundary_type = "pml"
            self.n_abs = n_abs
            self.rho_ext = np.asarray(rho_ext, dtype=np.float64)
            self.bulk_ext = np.asarray(bulk_ext,dtype=np.float64)
            self.sigma_x = np.asarray(sigma_x,dtype=np.float64)
            self.sigma_z = np.asarray(sigma_z,dtype=np.float64)
        else:
            raise ValueError("boundary_type must be 'dirichlet' or 'pml'.")

    # ========================================================
    # Forward modeling
    # ========================================================
    def forward(self,snapshot_times=None,):
        # ====================================================
        # Check source
        # ====================================================
        if self.wavelet is None:
            raise RuntimeError("Source has not been defined.")
        nt = len(self.wavelet)
        # ====================================================
        # Check receivers
        # ====================================================
        if self.ix_receivers is None:
            raise RuntimeError("Receivers have not been defined.")
        # ====================================================
        # Check boundary
        # ===================================================
        if self.boundary_type is None:
            raise RuntimeError("Boundary condition has not been defined.")
        # ====================================================
        # Snapshot times -> time steps
        # ====================================================
        if snapshot_times is None:
            snapshot_steps = np.empty(0, dtype=np.int64)
        else:
            snapshot_times = np.atleast_1d(np.asarray(snapshot_times, dtype=np.float64))
            snapshot_steps = np.rint(snapshot_times / self.dt).astype(np.int64)
            if np.any(snapshot_steps < 0) or np.any(snapshot_steps >= nt):
                raise ValueError("snapshot_times are outside the simulation time range.")
            if len(snapshot_steps) > 1:
                if np.any(np.diff(snapshot_steps) <= 0):
                    raise ValueError(
                        "snapshot_times must correspond to strictly increasing "
                        "and unique time steps."
                        )
        # ----------------------------------------------------
        # Check snapshot range
        # ----------------------------------------------------
        if np.any(snapshot_steps < 0) or np.any(snapshot_steps >= nt):
            raise ValueError("snapshot_times are outside the simulation time range.")
        # ====================================================
        # Select solver
        # ====================================================
        # ----------------------------------------------------
        # Dirichlet
        # ----------------------------------------------------
        if self.boundary_type == "dirichlet":
            seismogram, snapshots = (
                _acoustic_solver_dirichlet(
                    self.rho,
                    self.bulk,
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
            )
        # ----------------------------------------------------
        # PML
        # ----------------------------------------------------
        elif self.boundary_type == "pml":
            seismogram, snapshots = (
                _acoustic_solver_split_PML(
                    self.rho_ext,
                    self.bulk_ext,
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
            )
        else:
            raise RuntimeError(f"Unknown boundary type: "f"{self.boundary_type}")
        # ====================================================
        # Return
        # ====================================================
        return AcousticResult(
            snapshots=snapshots,
            seismogram=seismogram,
        )