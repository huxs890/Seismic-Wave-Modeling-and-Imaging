import numpy as np
from fd_coefficients import fd_coefficients
from PML_boundary import PML_boundary
from dataclasses import dataclass
from _acoustic_solver_dirichlet import _acoustic_solver_dirichlet
from _acoustic_solver_PML import _acoustic_solver_PML

@dataclass
class AcousticResult:
    #==========================================
    # Output of the second-order acoustic solver
    #==========================================
    snapshots: np.ndarray
    seismogram: np.ndarray

class SecondOrderAcousticForwardSolver:
    def __init__(
            self,
            velocity,
            dx,
            dz,
            dt,
            order=8,
    ):
        self.velocity = np.asarray(velocity,dtype=np.float64)
        self.dx = float(dx)
        self.dz = float(dz)
        self.dt = float(dt)
        self.order = int(order)
        self.nz,self.nx = self.velocity.shape

        # parameter checking
        # self._validate_model() 

        # FD coefficients
        self.coeff = np.asarray(fd_coefficients(self.order),dtype=np.float64)

        # Source
        self.wavelet = None
        self.ix_src = None
        self.iz_src = None

        # Receivers
        self.ix_receivers = None
        self.iz_receivers = None

        # Boundary condition
        self.boundary_type = None
        # PML parameters
        self.n_abs = None
        self.sigma = None
        self.velocity_ext = None

    def set_source(
              self,
              wavelet,
              ix,
              iz,
    ):
        wavelet = np.asarray(wavelet,dtype=np.float64)
        ix = int(ix)
        iz = int(iz)
        N = len(self.coeff)
        #------------------------------------------------------------------------
        # Check N
        #------------------------------------------------------------------------
        if ix < N or ix >= self.nx - N:   # index 怎么区分
            raise ValueError("Source x-position is too close to the boundary.")
        if iz < N or iz >= self.nz - N:
            raise ValueError("Source z-position is too close to the boundary.")
        #------------------------------------------------------------------------
        # Check wavelet
        #------------------------------------------------------------------------
        if wavelet.ndim != 1:
            raise ValueError("wavelet must be a 1-D array.")
        if len(wavelet) < 2:
            raise ValueError("wavelet must contain at least two time samples.")        
        self.wavelet = wavelet
        self.ix_src = ix
        self.iz_src = iz

    def set_receivers(
            self,
            ix,
            iz,
        ):
        ix = np.atleast_1d(np.asarray(ix, dtype=np.int64))
        iz = np.atleast_1d(np.asarray(iz, dtype=np.int64))
        #------------------------------------------------------------------------
        # Check if ix = iz
        #------------------------------------------------------------------------
        if len(ix) != len(iz):
            raise ValueError("ix and iz must have the same length.")
        #------------------------------------------------------------------------
        # Check another things
        #------------------------------------------------------------------------
        if ix.ndim != 1 or iz.ndim != 1:
            raise ValueError("Receiver indices must be 1-D arrays.")
        if ix.size == 0:
            raise ValueError("At least one receiver is required.")
        if np.any(ix < 0) or np.any(ix >= self.nx):
            raise ValueError("Receiver x-index is outside the model.")
        if np.any(iz < 0) or np.any(iz >= self.nz):
            raise ValueError("Receiver z-index is outside the model.")
        self.ix_receivers = ix
        self.iz_receivers = iz

    def set_boundary(
            self,
            boundary_type = 'PML',
            n_abs = 40,
            reflection = 1e-6,
            power=2,
    ):
        boundary_type = str(boundary_type).lower()
        #---------------------------------------------------------
        # Dirichlet boundary
        #---------------------------------------------------------
        if boundary_type == "dirichlet":
            self.boundary_type = "dirichlet"
            # PML-related parameters are not needed
            self.n_abs = None
            self.sigma = None
            self.velocity_ext = None
        #---------------------------------------------------------
        # PML boundary
        #---------------------------------------------------------
        elif boundary_type == "pml":
            self.boundary_type = "pml"
            # Check parameters
            n_abs = int(n_abs)
            if n_abs <= 0:
                raise ValueError("n_abs must be a positive integer.")
            if reflection <= 0.0 or reflection >= 1.0:
                raise ValueError("reflection must satisfy 0 < reflection < 1.")
            if power <= 0:
                raise ValueError("power must be positive.")
            # Generate extended velocity model and PML coefficients
            velocity_ext, sigma = PML_boundary(
                self.velocity,
                self.dx,
                self.dz,
                n_abs = n_abs,
                reflection = reflection,
                power = power,
            )
            self.boundary_type = 'pml'
            self.n_abs = n_abs
            self.velocity_ext = np.asarray(velocity_ext,dtype=np.float64)     
            self.sigma = np.asarray(sigma,dtype=np.float64)
        else:
            raise ValueError("boundary_type must be 'dirichlet' or 'pml'.")

    def forward(
            self,
            snapshot_times = None,
    ):
        #-------------------------------------------------------
        # Check source
        #-------------------------------------------------------
        if self.wavelet is None:
            raise RuntimeError("Source has not been defined.")
        nt = len(self.wavelet)
        #-------------------------------------------------------
        # Receivers
        #-------------------------------------------------------
        if self.ix_receivers is None:
            raise RuntimeError("Receivers have not been defined.")
        #-------------------------------------------------------
        # Check boundaries
        #-------------------------------------------------------        
        if self.boundary_type is None:
            raise RuntimeError("Boundary condition has not been defined.")
        #-------------------------------------------------------
        # Check boundaries
        #-------------------------------------------------------  
        if snapshot_times is None:
            snapshot_times = np.empty(0,dtype=np.float64)
        else:
            snapshot_times = np.asarray(snapshot_times,dtype=np.float64)
        snapshot_steps = np.rint(snapshot_times / self.dt).astype(np.int64)
        #-------------------------------------------------------
        # Check snapshot range
        #------------------------------------------------------- 
        if np.any(snapshot_steps < 0) or np.any(snapshot_steps >= nt):
             raise ValueError("snapshot_times are outside the simulation time range.")
        # Actual snapshot times
        actual_snapshot_times = snapshot_steps * self.dt
        #=======================================================
        # Select solver
        #=======================================================   
        if self.boundary_type == "dirichlet":
            seismogram, snapshots = _acoustic_solver_dirichlet(
                self.velocity,
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
        elif self.boundary_type == "pml":
            seismogram, snapshots = _acoustic_solver_PML(
                self.velocity_ext,
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
                self.sigma,
                self.n_abs,
            )
        else:
            raise RuntimeError(
            f"Unknown boundary type: "
            f"{self.boundary_type}"
        )
        #=======================================================
        # Return
        #=======================================================  
        return AcousticResult(
            snapshots=snapshots,
            seismogram=seismogram,
            )