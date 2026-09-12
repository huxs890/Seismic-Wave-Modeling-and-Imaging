"""Run: python -m unittest -v test_fwi (small, deterministic numerical tests)."""
import unittest
import numpy as np
from numba import set_num_threads
from boundary import damping_boundary
from fd_coefficients import fd_coefficients
from fwi_gradient import AcousticSetup, fwi_objective_gradient, simulate_shots
from fwi_optimization import parabolic_line_search, gradient_descent_fwi
from forward_acoustic_solver import forward_acoustic_solver


def small_problem():
    shape = (17, 19)
    z, x = np.indices(shape)
    v = 1950.0 + 3*x + 2*z
    truth = v + 90*np.exp(-((x-9)**2+(z-8)**2)/9)
    t = np.arange(151)*0.001
    a = np.pi*22*(t-0.035)
    wavelet = (1-2*a*a)*np.exp(-a*a)
    _, sigma = damping_boundary(np.full(shape, 2500.0), 10., 12., 6)
    setup = AcousticSetup(wavelet, fd_coefficients(4), np.array([4, 14]), np.array([2, 2]),
                          np.array([[2, 6, 10, 16], [2, 6, 10, 16]]),
                          np.full((2, 4), 14), 10., 12., .001, 6, sigma)
    return v, truth, setup, simulate_shots(truth, setup)


class TestFWI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(2)
        cls.v, cls.truth, cls.setup, cls.obs = small_problem()

    def test_discrete_gradient_including_padding(self):
        v, s, obs = self.v, self.setup, self.obs
        j, gm, g, _ = fwi_objective_gradient(v, obs, s)
        rng = np.random.default_rng(23)
        for label in ('random', 'edge', 'source'):
            p = rng.normal(size=v.shape)
            if label == 'edge':
                p[1:-1, 1:-1] = 0  # np.pad(mode=edge) chain rule must contribute.
            elif label == 'source':
                p[:] = 0
                p[s.iz_src, s.ix_src] = 1  # v^2 source amplitude derivative.
            p /= np.linalg.norm(p)
            analytic = float(np.sum(g*p))
            eps = .01
            jp = fwi_objective_gradient(v+eps*p, obs, s, False)[0]
            jm = fwi_objective_gradient(v-eps*p, obs, s, False)[0]
            numerical = (jp-jm)/(2*eps)
            error = abs(numerical-analytic)/max(abs(numerical), abs(analytic), 1e-15)
            print(f'gradient {label}: relative error={error:.3e}', flush=True)
            self.assertLess(error, 2e-6)
        np.testing.assert_allclose(gm, -.5*v**3*g)
        p = -g/np.max(np.abs(g))
        errors = []
        for eps in (1., .5, .25):
            value = fwi_objective_gradient(v+eps*p, obs, s, False)[0]
            errors.append(abs(value-j-eps*np.sum(g*p)))
        print('Taylor remainder ratios:', np.asarray(errors[:-1])/errors[1:], flush=True)
        self.assertGreater(errors[0]/errors[1], 3.7)
        self.assertGreater(errors[1]/errors[2], 3.7)

    def test_multishot_sum_and_zero_residual(self):
        s = self.setup
        j, _, g, _ = fwi_objective_gradient(self.v, self.obs, s)
        total_j, total_g = 0., np.zeros_like(g)
        for i in range(2):
            shot = AcousticSetup(s.wavelet, s.coeff, s.ix_src[i:i+1], s.iz_src[i:i+1],
                                 s.ix_receiver[i:i+1], s.iz_receiver[i:i+1],
                                 s.dx, s.dz, s.dt, s.n_abs, s.sigma)
            ji, _, gi, _ = fwi_objective_gradient(self.v, self.obs[i:i+1], shot)
            total_j += ji
            total_g += gi
        self.assertAlmostEqual(total_j, j)
        np.testing.assert_allclose(total_g, g, rtol=1e-12, atol=1e-15)
        j0, _, g0, _ = fwi_objective_gradient(self.truth, self.obs, s)
        self.assertEqual(j0, 0.)
        np.testing.assert_array_equal(g0, 0.)

    def test_parabola_and_rejection(self):
        r = parabolic_line_search(lambda a: (a-3)**2+1, 10, -6, 1, 10)
        self.assertAlmostEqual(r.alpha, 3., places=10)
        self.assertAlmostEqual(r.objective, 1.)
        failed = parabolic_line_search(lambda a: 1+a, 1, -1, 1, 10, 4)
        self.assertEqual(failed.alpha, 0.)
        limited = parabolic_line_search(lambda a: (a-30)**2, 900, -60, 1, 2)
        self.assertEqual(limited.alpha, 2.)

    def test_iterations_and_bounds(self):
        mask = np.ones_like(self.v, dtype=bool)
        mask[:2] = False
        r = gradient_descent_fwi(self.v, self.obs, self.setup, max_iterations=3,
                                 initial_step=10, max_step=60, velocity_bounds=(1900, 2150),
                                 update_mask=mask, relative_tolerance=0, max_line_evaluations=6)
        self.assertEqual(r['iterations'], 3)
        self.assertTrue(np.all(np.diff(r['objective_history']) < 0))
        self.assertTrue(np.all((r['velocity'] >= 1900) & (r['velocity'] <= 2150)))
        np.testing.assert_array_equal(r['velocity'][:2], self.v[:2])
        recomputed = fwi_objective_gradient(r['velocity'], self.obs, self.setup, False)[0]
        self.assertAlmostEqual(recomputed, r['objective_history'][-1], places=12)
        self.assertLess(np.linalg.norm(r['velocity']-self.truth), np.linalg.norm(self.v-self.truth))
        zeros = gradient_descent_fwi(self.truth, self.obs, self.setup, max_iterations=2,
                                     velocity_bounds=(1900, 2400), verbose=False)
        self.assertEqual(zeros['iterations'], 0)

    def test_invalid_geometry_and_cfl(self):
        with self.assertRaises(ValueError):
            fwi_objective_gradient(self.v, self.obs[0], self.setup)
        with self.assertRaises(ValueError):
            self.setup.validate_velocity(np.full_like(self.v, 100000))

    def test_first_and_last_time_samples_and_duplicate_receivers(self):
        # Very short window isolates t=0 source derivative and terminal residual.
        v = np.full((5, 5), 2000.)
        _, sigma = damping_boundary(v, 10., 10., 4)
        setup = AcousticSetup(np.array([1., 2., 99.]), fd_coefficients(8),
                              np.array([2]), np.array([2]), np.array([[2, 2]]),
                              np.array([[2, 2]]), 10., 10., .001, 4, sigma)
        obs = simulate_shots(v+20, setup)
        _, _, g, syn = fwi_objective_gradient(v, obs, setup)
        p = np.zeros_like(v)
        p[2, 2] = 1.
        eps = .01
        jp = fwi_objective_gradient(v+eps*p, obs, setup, False)[0]
        jm = fwi_objective_gradient(v-eps*p, obs, setup, False)[0]
        np.testing.assert_allclose(g[2, 2], (jp-jm)/(2*eps), rtol=1e-7)
        np.testing.assert_array_equal(syn[:, :, 0], syn[:, :, 1])
        # wavelet[-1] is outside the nt-1 updates and must not affect recorded u[nt-1].
        setup.wavelet[-1] = -999.
        np.testing.assert_array_equal(simulate_shots(v, setup), syn)

    def test_original_forward_interface(self):
        s, v = self.setup, self.v
        ve = np.pad(v, s.n_abs, mode='edge')
        snapshots, data, utt = forward_acoustic_solver(
            ve, s.wavelet, s.coeff, s.ix_src[0], s.iz_src[0], s.ix_receiver[0],
            s.iz_receiver[0], s.dx, s.dz, s.dt, np.array([1, 10]), s.sigma, s.n_abs,
        )
        _, lean_data, empty_history = s.shot(ve, 0, False)
        _, grad_data, sensitivity = s.shot(ve, 0, True)
        # Numba specializes omitted defaults separately; fastmath can reorder
        # arithmetic at roundoff level (observed max difference 3.2e-14).
        np.testing.assert_allclose(data, lean_data, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(data, grad_data, rtol=1e-12, atol=1e-13)
        self.assertEqual(empty_history.size, 0)
        self.assertEqual(snapshots.shape, (2, *v.shape))
        physical = sensitivity[:, s.n_abs:-s.n_abs, s.n_abs:-s.n_abs]
        np.testing.assert_allclose(physical, 2*s.dt**2*utt[:-1]/v, rtol=1e-10, atol=1e-13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
