"""Scale-separated cubic-spline broadband, averaged over measured mode bins.

The kernel follows Chen et al. (2024), equations 29-30, applied separately
to the five measured mu wedges. This is not their galaxy physical model.
The primary knot spacing is .06 h/Mpc; .05 and .08 are sensitivity choices.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from fit_binned_pk import BinnedPkModel

PARS = ('ap', 'at', 'beta', 'sig_par', 'sig_per')
LOW = np.array([.8, .8, 0., 0., 0.])
HIGH = np.array([1.2, 1.2, 5., 20., 20.])


def cubic_kernel(x):
    x = np.abs(np.asarray(x))
    return np.where(x <= 1, (4-6*x*x+3*x**3)/6,
                    np.where(x < 2, (2-x)**3/6, 0.))


class SplinePkModel(BinnedPkModel):
    def __init__(self, cosmo, rule, mask, spacing=.06, peak_only=True):
        super().__init__(cosmo, rule, mask, nbb=3, peak_only=peak_only)
        self.spacing = spacing
        k = rule.k[mask]
        self.centers = np.arange(-1, int(np.ceil(k.max()/spacing))+2)
        columns = []
        for wedge in range(5):
            for center in self.centers:
                column = (self.iw == wedge)*np.sum(
                    self.weights*cubic_kernel(k/spacing-center), axis=1)
                if np.any(column):
                    columns.append(column)
        self.BB = np.column_stack(columns)


class ProjectedFitter:
    """Profile the linear nuisance space with one cached weighted SVD.

This is ordinary weighted least squares, not regularization or a prior
favoring alpha=1. All nonlinear parameters are free within explicit bounds.
"""
    def __init__(self, model, data, ivar):
        self.model = model
        self.sqrt_weight = np.sqrt(np.asarray(ivar))
        weighted = model.BB*self.sqrt_weight[:, None]
        norm = np.linalg.norm(weighted, axis=0)
        weighted = weighted[:, norm > 0]/norm[norm > 0]
        u, s, _ = np.linalg.svd(weighted, full_matrices=False)
        self.rank = int(np.sum(s > s[0]*1e-11))
        self.q = u[:, :self.rank]
        self.data = self.project(np.asarray(data)*self.sqrt_weight)

    def project(self, vector):
        return vector-self.q@(self.q.T@vector)

    def residual(self, theta):
        signal = self.project(self.model.peak(*theta)*self.sqrt_weight)
        norm = signal@signal
        if norm <= 0:
            raise ValueError('No template information remains after projection')
        amplitude = (signal@self.data)/norm
        return self.data-amplitude*signal

    def fit(self, start=(1., 1., 1., 4., 2.), multistart=True):
        begin = time.process_time()
        starts = [np.clip(np.asarray(start), LOW+1e-7, HIGH-1e-7)]
        if multistart:
            starts.append(np.array([1., 1., 1.5, 6., 3.]))
        results = [least_squares(self.residual, x, bounds=(LOW, HIGH), jac='3-point',
                    x_scale=[.01, .01, 1., 3., 3.], ftol=1e-11, xtol=1e-11,
                    gtol=1e-8, max_nfev=1200) for x in starts]
        result = min(results, key=lambda r: r.fun@r.fun)
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f'Failed fit: {result.message}')
        signal = self.project(self.model.peak(*result.x)*self.sqrt_weight)
        amplitude = (signal@self.data)/(signal@signal)
        out = dict(zip(PARS, result.x))
        out.update(chi2=float(result.fun@result.fun), bsq=amplitude, valid=True,
                   rank=self.rank, active_bounds=np.array(result.active_mask),
                   near_bounds=np.minimum(result.x-LOW, HIGH-result.x)<1e-3,
                   start_disagreement=max(np.max(abs(r.x[:2]-result.x[:2])) for r in results),
                   nfev=sum(r.nfev for r in results), cpu_seconds=time.process_time()-begin)
        return out

    def derivative_retention(self, theta):
        """Fraction of squared scale derivatives retained after BB projection.

        This is a diagnostic of signal removal, not a marginalized forecast.
        """
        ratios = []
        for j in [0, 1]:
            plus = np.array(theta, dtype=float); minus = plus.copy()
            plus[j] += 1e-4; minus[j] -= 1e-4
            derivative = (self.model.peak(*plus)-self.model.peak(*minus))*self.sqrt_weight
            projected = self.project(derivative)
            ratios.append((projected@projected)/(derivative@derivative))
        return np.array(ratios)
