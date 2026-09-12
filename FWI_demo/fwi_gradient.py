import numpy as np
from dataclasses import dataclass
from numba import njit, prange
from forward_acoustic_solver import forward_acoustic_solver

def fwi_gradient(
        wavefield_tt, 
        lambda_wavefield, 
        velocity,
        dt
):
    # Basic parameters
    nt, nz, nx = wavefield_tt.shape
    # Initialize gradient
    gradient_m = np.zeros((nz, nx),dtype=np.float64)
    # ===========================================
    # Time-domain zero-lag cross-correlation
    # ===========================================
    # 旧单炮接口，t=0 的非零震源项也必须参与相关。
    # 此接口仅为物理域连续伴随近似；完整反演使用下面的离散伴随接口。
    for it in range(nt - 1):
        # Forward wavefield second time derivative
        utt = wavefield_tt[it]
        # Adjoint wavefield at the SAME physical time
        lam = lambda_wavefield[nt-1-it]
        # Zero-lag cross-correlation    
        gradient_m[:, :] -= dt * utt * lam

    gradient_v = -2.0* gradient_m / velocity**3

    return gradient_m,gradient_v

# 多炮求和、边界链式求导和数据目标函数统一放在梯度模块中。
@dataclass
class AcousticSetup:
    """Fixed acquisition/discretization for J = dt/2 * sum((d_syn-d_obs)**2).

    sigma must be held fixed for data generation, all iterates and line search.
    Its derivative is deliberately zero. Velocity uses edge padding each call.
    Wavefields remain float64; shot wavefields are processed one at a time.
    """
    wavelet: np.ndarray
    coeff: np.ndarray
    ix_src: np.ndarray
    iz_src: np.ndarray
    ix_receiver: np.ndarray
    iz_receiver: np.ndarray
    dx: float
    dz: float
    dt: float
    n_abs: int
    sigma: np.ndarray

    def __post_init__(self):
        for name in ('wavelet', 'coeff', 'sigma'):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be finite')
            setattr(self, name, np.ascontiguousarray(value.copy()))
        for name in ('ix_src', 'iz_src', 'ix_receiver', 'iz_receiver'):
            value = np.asarray(getattr(self, name))
            if not np.all(np.isfinite(value)) or not np.all(value == np.floor(value)):
                raise ValueError(f'{name} must contain integer indices')
            setattr(self, name, np.ascontiguousarray(value, dtype=np.int64))
        if self.wavelet.ndim != 1 or self.wavelet.size < 3:
            raise ValueError('wavelet must be 1D with at least 3 time samples')
        if self.coeff.ndim != 1 or not self.coeff.size or not np.any(self.coeff):
            raise ValueError('coeff must be a nonzero 1D stencil')
        if not all(np.isfinite(x) and x > 0 for x in (self.dx, self.dz, self.dt)):
            raise ValueError('dx, dz and dt must be finite and positive')
        if not isinstance(self.n_abs, (int, np.integer)) or self.n_abs < len(self.coeff):
            raise ValueError('n_abs must be an integer at least the stencil half-width')
        if self.sigma.ndim != 2 or np.any(self.sigma < 0):
            raise ValueError('sigma must be a nonnegative 2D field')
        if self.ix_src.ndim != 1 or not self.ix_src.size or self.iz_src.shape != self.ix_src.shape:
            raise ValueError('source arrays must be matching nonempty 1D arrays')
        if (self.ix_receiver.ndim != 2 or self.ix_receiver.shape[0] != self.ix_src.size
                or self.ix_receiver.shape[1] == 0 or self.iz_receiver.shape != self.ix_receiver.shape):
            raise ValueError('receiver arrays must have shape (n_shot, n_receiver)')
        nz, nx = self.model_shape
        if min(nz, nx) <= 0:
            raise ValueError('sigma shape must contain a nonempty physical domain')
        for name, limit in (('ix_src', nx), ('iz_src', nz), ('ix_receiver', nx), ('iz_receiver', nz)):
            if np.any(getattr(self, name) < 0) or np.any(getattr(self, name) >= limit):
                raise ValueError(f'{name} outside physical model')

    @property
    def model_shape(self):
        return tuple(n - 2*self.n_abs for n in self.sigma.shape)

    @property
    def data_shape(self):
        return (self.ix_src.size, self.wavelet.size, self.ix_receiver.shape[1])

    @property
    def stable_velocity_limit(self):
        # Conservative spectral bound: dt^2*v^2*rho(-L) < 4.
        return 1.0 / (self.dt*np.sqrt(np.sum(np.abs(self.coeff))*(self.dx**-2+self.dz**-2)))

    def validate_velocity(self, velocity):
        v = np.ascontiguousarray(velocity, dtype=np.float64)
        if v.shape != self.model_shape or not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError('velocity must match the physical domain and be finite, positive')
        if np.max(v) >= self.stable_velocity_limit:
            raise ValueError('velocity exceeds conservative CFL bound; reduce dt')
        return v

    def shot(self, velocity_ext, i, need_gradient=False):
        return forward_acoustic_solver(
            velocity_ext, self.wavelet, self.coeff, self.ix_src[i], self.iz_src[i],
            self.ix_receiver[i], self.iz_receiver[i], self.dx, self.dz, self.dt,
            np.empty(0, dtype=np.int64), self.sigma, self.n_abs,
            False, need_gradient, False,
        )


@njit(parallel=True)
def _discrete_adjoint_gradient(velocity_ext, sensitivity, residual, coeff,
                               ix_receiver, iz_receiver, dx, dz, dt, sigma, n_abs):
    """Reverse differentiation of the actual forward recurrence (including source).

    A = diag(2/d) + diag(v^2*dt^2/d)*L, B = diag(-(1-sigma*dt)/d).
    Apply A.T = diag(2/d) + L.T*diag(v^2*dt^2/d), not A.
    residual[t] samples u[t]; inject it BEFORE correlating with step t-1.
    """
    nz, nx = velocity_ext.shape
    N = len(coeff)
    d = 1.0 + sigma*dt
    scale = velocity_ext**2*dt**2/d
    b = -(1.0-sigma*dt)/d
    q = np.zeros((nz, nx))
    future = np.zeros((nz, nx))
    previous = np.zeros((nz, nx))
    weighted = np.zeros((nz, nx))
    gradient = np.zeros((nz, nx))
    for it in range(residual.shape[0]-2, -1, -1):
        # Serial injection correctly accumulates even coincident receiver traces.
        for ir in range(len(ix_receiver)):
            q[iz_receiver[ir]+n_abs, ix_receiver[ir]+n_abs] += dt*residual[it+1, ir]
        for iz in prange(N, nz-N):
            for ix in range(N, nx-N):
                gradient[iz, ix] += q[iz, ix]*sensitivity[it, iz, ix]
                weighted[iz, ix] = scale[iz, ix]*q[iz, ix]
        for iz in prange(N, nz-N):
            for ix in range(N, nx-N):
                lap = 0.0
                for m in range(N):
                    k = m+1
                    lap += coeff[m]*((weighted[iz, ix+k]+weighted[iz, ix-k]-2*weighted[iz, ix])/dx**2
                                   +(weighted[iz+k, ix]+weighted[iz-k, ix]-2*weighted[iz, ix])/dz**2)
                previous[iz, ix] = 2*q[iz, ix]/d[iz, ix] + lap + b[iz, ix]*future[iz, ix]
        future, q, previous = q, previous, future
    return gradient

def fwi_objective_gradient(velocity, d_obs, setup, compute_gradient=True):
    """Return (J, gradient_m, gradient_v, d_syn); sum ALL shots here.

    m=1/v^2; gradient_m=-v^3/2*gradient_v (Euclidean grid derivatives).
    No dx*dz multiplier because the specified data objective is dt/2*sum(r^2).
    When compute_gradient=False, both gradients are None and no history is saved.
    Memory scales with one shot, not n_shot. sigma is never recomputed here.
    """
    v = setup.validate_velocity(velocity)
    observed = np.asarray(d_obs, dtype=np.float64)
    if observed.shape != setup.data_shape or not np.all(np.isfinite(observed)):
        raise ValueError('d_obs must be finite and match (n_shot, nt, n_receiver)')
    ve = np.pad(v, setup.n_abs, mode='edge')
    synthetic = np.empty(setup.data_shape, dtype=np.float64)
    gradient_ext = np.zeros_like(ve) if compute_gradient else None
    objective = 0.0
    for i in range(setup.ix_src.size):
        _, synthetic[i], sensitivity = setup.shot(ve, i, compute_gradient)
        residual = synthetic[i] - observed[i]
        objective += 0.5*setup.dt*np.sum(residual**2)
        if compute_gradient:
            gradient_ext += _discrete_adjoint_gradient(
                ve, sensitivity, residual, setup.coeff, setup.ix_receiver[i], setup.iz_receiver[i],
                setup.dx, setup.dz, setup.dt, setup.sigma, setup.n_abs)
        del sensitivity
    if not np.isfinite(objective):
        raise FloatingPointError('non-finite objective')
    if not compute_gradient:
        return float(objective), None, None, synthetic
    # FIX: transpose of np.pad(mode='edge'): fold pad derivatives onto edge cells.
    nz, nx = v.shape
    zz = np.clip(np.arange(ve.shape[0])-setup.n_abs, 0, nz-1)
    xx = np.clip(np.arange(ve.shape[1])-setup.n_abs, 0, nx-1)
    gradient_v = np.zeros_like(v)
    np.add.at(gradient_v, (zz[:, None], xx[None, :]), gradient_ext)
    if not np.all(np.isfinite(gradient_v)):
        raise FloatingPointError('non-finite gradient')
    return float(objective), -0.5*v**3*gradient_v, gradient_v, synthetic


def simulate_shots(velocity, setup):
    """Forward data only, with the same fixed sigma as the inversion."""
    v = setup.validate_velocity(velocity)
    ve = np.pad(v, setup.n_abs, mode='edge')
    data = np.empty(setup.data_shape, dtype=np.float64)
    for i in range(setup.ix_src.size):
        data[i] = setup.shot(ve, i)[1]
    return data
