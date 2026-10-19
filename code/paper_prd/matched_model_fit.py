"""Common physical template and transformable spline broadband in P and xi.

Both models use the IC-derived peak-only template, no Arinyo correction,
and the same observed-coordinate compact filter. Finite fitting cuts and
the weights still differ. These are sensitivity tests, not identical
likelihoods or independent determinations of a physical shift.
"""
from pathlib import Path
import numpy as np
from scipy.special import eval_legendre, spherical_jn, roots_legendre
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_triangular
from target_bandpass import pass_power, save_atomic

OUT = Path('/pscratch/sd/b/boryanah/Lyaf_recon/matched_model_20260909')
SPACING = .06
ELLS = (0, 2, 4, 6)
CENTERS = np.arange(-1, int(np.ceil(.5/SPACING))+2)


def cubic_kernel(x):
    x = np.abs(np.asarray(x))
    return np.where(x <= 1, (4-6*x*x+3*x**3)/6,
                    np.where(x < 2, (2-x)**3/6, 0.))


class MatchedPkModel:
    def __init__(self, cosmo, rule, mask):
        k, mu = rule.k[mask], rule.mu[mask]
        self.cosmo = cosmo; self.k = k; self.mu = mu
        self.weights = rule.weight[mask]*pass_power(k, .35)
        self.BB = np.column_stack([
            np.sum(self.weights*eval_legendre(ell, mu)*cubic_kernel(k/SPACING-n), axis=1)
            for ell in ELLS for n in CENTERS])

    def peak(self, ap, at, beta, sp, st):
        kp = self.k*self.mu/ap
        kt = self.k*np.sqrt(np.maximum(1-self.mu**2, 0))/at
        k = np.hypot(kp, kt); mu = kp/k
        damping = np.exp(-.5*((kp*sp)**2+(kt*st)**2))
        wiggle = self.cosmo.plin(k)-self.cosmo.plin(k, wiggle=False)
        p = (1+beta*mu**2)**2*wiggle*damping/(ap*at**2)
        p += (1+beta*self.mu**2)**2*self.cosmo.plin(self.k, wiggle=False)
        return np.sum(self.weights*p, axis=1)


def xi_broadband(rule, order=10, dr=.2):
    """Hankel-transform the same Fourier basis, then average each xi bin.

Split quadrature at every spline break and filter break, including endpoints.
The i**ell sign and 1/(2 pi**2) normalization use P(k) in (Mpc/h)^3.
"""
    breaks = np.unique(np.r_[np.arange(101)*.005,
                            np.arange(10)*SPACING, .35, .5])
    breaks = breaks[(breaks >= 0) & (breaks <= .5)]
    g, w = roots_legendre(order)
    lo, hi = breaks[:-1, None], breaks[1:, None]
    k = ((lo+hi)/2+(hi-lo)*g/2).ravel()
    weight = ((hi-lo)*w/2).ravel()*k*k*pass_power(k, .35)/(2*np.pi**2)
    b = cubic_kernel(k[:, None]/SPACING-CENTERS[None, :])
    r = np.hypot(rule.rp, rule.rt); mu = rule.rp/r
    grid = np.arange(0, np.max(r)+2*dr, dr)
    columns = []
    for ell in ELLS:
        transform = (-1)**(ell//2)*(spherical_jn(ell, grid[:, None]*k[None, :])@(weight[:, None]*b))
        values = CubicSpline(grid, transform)(r)*eval_legendre(ell, mu)[:, None]
        columns.extend(rule.average(values[:, j]) for j in range(len(CENTERS)))
    return np.column_stack(columns)


def xi_fitter():
    # Deferred import keeps measurement/Fourier jobs independent of Vega.
    from fit_common_target import CommonTargetFitter
    fitter = CommonTargetFitter(512, .35, small_scale_nl='none', linear_template='ic_savgol')
    path = OUT/'xi_spline_basis.npz'
    if path.exists():
        with np.load(path) as d:
            basis = d['basis']; np.testing.assert_array_equal(d['mask'], fitter.mask)
    else:
        basis = xi_broadband(fitter.rule)
        fine = xi_broadband(fitter.rule, order=14, dr=.1)
        # A vector norm avoids meaningless fractional tests at Bessel zeros.
        error = np.linalg.norm(basis-fine, axis=0)/np.maximum(np.linalg.norm(fine, axis=0), 1e-30)
        assert error.max() < 2e-5, error
        save_atomic(path, basis=basis, mask=fitter.mask, convergence=error,
                    spacing=SPACING, ells=np.array(ELLS), centers=CENTERS)
    # Whiten and orthogonalize, without a nuisance prior or alpha prior.
    chol = np.linalg.cholesky(fitter.icov)
    weighted = chol.T@basis
    norms = np.linalg.norm(weighted, axis=0)
    weighted = weighted[:, norms > 0]/norms[norms > 0]
    u, s, _ = np.linalg.svd(weighted, full_matrices=False)
    rank = np.sum(s > s[0]*1e-11)
    fitter.basis = solve_triangular(chol.T, u[:, :rank], lower=False)
    fitter.icov_b = fitter.icov@fitter.basis
    fitter.normal = fitter.basis.T@fitter.icov_b
    fitter.broadband_rank = int(rank)
    fitter.broadband_singular_values = s
    # Match the Fourier nuisance bounds, in addition to the physical waveform.
    fitter.sample['limits']['ap'] = (.8, 1.2)
    fitter.sample['limits']['at'] = (.8, 1.2)
    fitter.sample['limits']['beta_LYA'] = (0., 5.)
    fitter.sample['limits']['sigmaNL_par'] = (0., 20.)
    fitter.sample['limits']['sigmaNL_per'] = (0., 20.)
    return fitter
