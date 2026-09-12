"""FIX/NEW: gradient descent with safeguarded three-point parabolic line search."""
from dataclasses import dataclass
import numpy as np
from fwi_gradient import fwi_objective_gradient


@dataclass
class LineSearchResult:
    alpha: float
    objective: float
    evaluations: list
    status: str


def parabolic_line_search(objective, f0, slope0, initial_step=10.0,
                          max_step=80.0, max_evaluations=12):
    """Minimize phi(alpha), alpha>=0, using an evaluated three-point parabola.

    First halve/double trial steps to bracket a minimum. Fit a parabola to
    (a,phi(a)), (b,phi(b)), (c,phi(c)); evaluate its vertex, update the bracket.
    Degenerate curvature/outside vertices use interval contraction. Only a
    real, evaluated point satisfying strict decrease and Armijo is accepted.
    A boundary minimum can be accepted when the velocity/step cap is reached.
    """
    if not np.isfinite(f0) or not np.isfinite(slope0) or slope0 >= 0:
        raise ValueError('finite f0 and a negative descent slope are required')
    if (not np.isfinite(initial_step) or not np.isfinite(max_step)
            or initial_step <= 0 or max_step <= 0):
        raise ValueError('initial_step and max_step must be finite, positive')
    if not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 3:
        raise ValueError('max_evaluations must be an integer >= 3')
    samples = [(0.0, float(f0))]

    def evaluate(alpha):
        f = float(objective(float(alpha)))
        if not np.isfinite(f):
            f = float('inf')
        samples.append((float(alpha), f))
        return f

    def accepted(alpha, f):
        return alpha > 0 and f < f0 and f <= f0 + 1e-4*alpha*slope0

    def finish(status):
        candidates = [(a, f) for a, f in samples if accepted(a, f)]
        if not candidates:
            return LineSearchResult(0.0, float(f0), samples, 'no_decrease')
        alpha, value = min(candidates, key=lambda pair: pair[1])
        return LineSearchResult(alpha, value, samples, status)

    a, fa = 0.0, float(f0)
    b = min(float(initial_step), float(max_step))
    fb = evaluate(b)
    c = fc = None
    while not accepted(b, fb) and len(samples)-1 < max_evaluations:
        c, fc = b, fb
        b *= 0.5
        fb = evaluate(b)
    if not accepted(b, fb):
        return finish('no_decrease')
    if c is None:
        while len(samples)-1 < max_evaluations:
            if b >= max_step:
                return finish('step_cap')
            c = min(2*b, max_step)
            fc = evaluate(c)
            if fc >= fb:
                break
            a, fa, b, fb = b, fb, c, fc
            c = fc = None
        if c is None:
            return finish('evaluation_limit')
    # Bracket invariant: a < b < c, f(b) <= f(a), f(b) <= f(c).
    while len(samples)-1 < max_evaluations:
        width = c-a
        if width <= 1e-5*max(1.0, abs(b)):
            break
        s_ab = (fb-fa)/(b-a)
        s_bc = (fc-fb)/(c-b)
        curvature = (s_bc-s_ab)/(c-a)
        vertex = (a+b)/2-s_ab/(2*curvature) if np.isfinite(curvature) and curvature > 0 else np.nan
        # Safeguard against extrapolation or repeat evaluations.
        margin = 1e-3*width
        if (not np.isfinite(vertex) or vertex <= a+margin or vertex >= c-margin
                or abs(vertex-b) < margin):
            vertex = (a+b)/2 if b-a > c-b else (b+c)/2
        fv = evaluate(vertex)
        if vertex < b:
            if fv < fb:
                c, fc, b, fb = b, fb, vertex, fv
            else:
                a, fa = vertex, fv
        else:
            if fv < fb:
                a, fa, b, fb = b, fb, vertex, fv
            else:
                c, fc = vertex, fv
    return finish('parabolic')


def gradient_descent_fwi(velocity_initial, d_obs, setup, max_iterations=10,
                         initial_step=10.0, max_step=80.0,
                         velocity_bounds=(1500.0, 3500.0), update_mask=None,
                         relative_tolerance=1e-5, gradient_tolerance=0.0,
                         max_line_evaluations=10, verbose=True):
    """Bound-constrained steepest descent, v_next = v + alpha*p.

    p = -projected_gradient / max(abs(projected_gradient)); alpha is in m/s
    and bounds the largest cell update. This positive scalar normalization
    preserves the steepest-descent direction; raw step = alpha / max|g|.
    The feasible step is computed BEFORE the line search (no clipped parabola).
    History contains the initial J followed ONLY by accepted updates.
    """
    v = setup.validate_velocity(velocity_initial).copy()
    lo, hi = map(float, velocity_bounds)
    if not np.isfinite(lo) or not np.isfinite(hi) or not 0 < lo < hi:
        raise ValueError('velocity_bounds must be finite, positive and increasing')
    # CFL also limits every evaluated trial model.
    hi = min(hi, 0.999*setup.stable_velocity_limit)
    if hi <= lo or np.any(v < lo) or np.any(v > hi):
        raise ValueError('initial model outside velocity/CFL bounds')
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer')
    if (not np.isfinite(relative_tolerance) or relative_tolerance < 0
            or not np.isfinite(gradient_tolerance) or gradient_tolerance < 0):
        raise ValueError('stopping tolerances must be finite and nonnegative')
    if not np.isfinite(initial_step) or not np.isfinite(max_step) or min(initial_step, max_step) <= 0:
        raise ValueError('step sizes must be finite and positive')
    if not isinstance(max_line_evaluations, (int, np.integer)) or max_line_evaluations < 3:
        raise ValueError('max_line_evaluations must be an integer >= 3')
    mask = np.ones(v.shape, dtype=bool) if update_mask is None else np.asarray(update_mask, dtype=bool)
    if mask.shape != v.shape:
        raise ValueError('update_mask must match velocity')
    j, _, g, _ = fwi_objective_gradient(v, d_obs, setup)
    history, models, steps, raw_steps, searches = [j], [v.copy()], [], [], []
    status = 'max_iterations'
    for iteration in range(max_iterations):
        projected = g.copy()
        projected[~mask] = 0.0
        projected[(v <= lo) & (projected > 0)] = 0.0
        projected[(v >= hi) & (projected < 0)] = 0.0
        norm = float(np.max(np.abs(projected)))
        if norm <= gradient_tolerance:
            status = 'projected_gradient_zero'
            break
        p = -projected/norm
        slope = float(np.sum(g*p))
        feasible = float(max_step)
        if np.any(p > 0):
            feasible = min(feasible, float(np.min((hi-v[p > 0])/p[p > 0])))
        if np.any(p < 0):
            feasible = min(feasible, float(np.min((lo-v[p < 0])/p[p < 0])))
        if feasible <= 0:
            status = 'bound_stall'
            break

        def phi(alpha):
            return fwi_objective_gradient(v+alpha*p, d_obs, setup, compute_gradient=False)[0]

        search = parabolic_line_search(phi, j, slope, initial_step,
                                       feasible, max_line_evaluations)
        searches.append(search)
        if search.alpha == 0:
            status = 'line_search_failed'
            break
        updated = v+search.alpha*p
        if np.array_equal(updated, v):
            status = 'roundoff_stall'
            break
        previous = j
        v, j = updated, search.objective
        history.append(j)
        models.append(v.copy())
        steps.append(search.alpha)
        raw_steps.append(search.alpha/norm)
        if verbose:
            print(f'Iteration {iteration+1:02d}: J={j:.8e}, J/J0={j/history[0]:.6f}, '
                  f'step={search.alpha:.5g} m/s, search={search.status}', flush=True)
        if (previous-j)/max(abs(previous), np.finfo(float).tiny) <= relative_tolerance:
            status = 'relative_decrease'
            break
        if iteration+1 < max_iterations:
            j, _, g, _ = fwi_objective_gradient(v, d_obs, setup)
    return {'velocity': v, 'objective_history': np.asarray(history),
            'velocity_history': np.asarray(models), 'step_history': np.asarray(steps),
            'raw_step_history': np.asarray(raw_steps), 'line_searches': searches,
            'status': status, 'iterations': len(steps), 'effective_bounds': (lo, hi)}
