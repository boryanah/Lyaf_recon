#!/usr/bin/env python
"""
fit_bao_apat_standalone.py
--------------------------
Self-contained (numpy-only) estimator of the BAO dilation parameters
alpha_parallel (ap) and alpha_transverse (at) from xi(r_perp, r_par) grids,
and of the pre- vs post-reconstruction improvement in their uncertainties.

No vega / picca / cosmoprimo / scipy. Everything is implemented here:

  * Linear P(k): Eisenstein & Hu (1998) transfer function, full (wiggle)
    and no-wiggle versions, for the AbacusSummit c000 (Planck 2018) cosmology.
  * Redshift-space template:
        P(k,mu) = bsq * (1 + beta*mu^2)^2 *
                  [ (P_lin - P_nw) * exp(-k^2(mu^2 Sig_par^2 +
                                          (1-mu^2) Sig_perp^2)/2) + P_nw ]
    -> multipoles P_0, P_2, P_4 by Gauss-Legendre -> xi_0, xi_2, xi_4 by
    spherical Bessel transforms -> xi(r_perp, r_par).
  * AP dilation applied to the model:  xi_mod(rt, rl) = xi_tpl(at*rt, ap*rl).
  * Smooth broadband nuisance terms (linear, profiled analytically).
  * Nelder-Mead minimizer (hand-rolled).
  * Errors: (a) scatter of best-fit ap/at across realizations,
            (b) Delta chi^2 = 1 profile around the minimum for the mean fit.

Input data format
-----------------
npz files with keys 'rp_bins', 'pi_bins', 'xirppi'  (the format written by
bin_corr_fft.py; 'rp' = transverse, 'pi' = line of sight). If xirppi has a
leading axis of length 3 (LCV outputs), the last entry is used.

Usage
-----
  # self-test on synthetic data (recommended first run):
  python fit_bao_apat_standalone.py --selftest

  # real comparison:
  python fit_bao_apat_standalone.py \
      --prerecon  data_fft/Xi_rppi_LyAxLyA_ph000_....npz [more ...] \
      --postrecon data_recon/Xi_rppi_LyAxLyA_recon_ph000_....npz [more ...] \
      --rmin 30 --rmax 180

Author's note: alpha conventions follow the standard Lya BAO ones:
ap scales line-of-sight separations, at scales transverse separations,
so ap = at = 1 when the data matches the fiducial (template) cosmology.
"""

import argparse
import sys

import numpy as np

# numpy 1.x / 2.x compatibility (np.trapz was renamed to np.trapezoid)
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# ═════════════════════════════════════════════════════════════════════════════
# 1. Linear power spectrum: Eisenstein & Hu (1998), wiggle and no-wiggle
#    (astro-ph/9709112; all fitting formulas transcribed from the paper)
# ═════════════════════════════════════════════════════════════════════════════

class EH98:
    """EH98 transfer functions for a flat LCDM cosmology. k in h/Mpc."""

    def __init__(self, h=0.6736, ombh2=0.02237, omch2=0.1200,
                 ns=0.9649, Tcmb=2.7255):
        self.h, self.ns = h, ns
        wm = ombh2 + omch2          # omega_m h^2
        wb = ombh2
        self.wm, self.wb = wm, wb
        self.Om0 = wm / h**2
        th = Tcmb / 2.7             # Theta_2.7
        self.th = th

        # ── scales ───────────────────────────────────────────────────────────
        self.zeq = 2.50e4 * wm * th**-4
        self.keq = 7.46e-2 * wm * th**-2                       # 1/Mpc
        b1 = 0.313 * wm**-0.419 * (1. + 0.607 * wm**0.674)
        b2 = 0.238 * wm**0.223
        self.zd = 1291. * wm**0.251 / (1. + 0.659 * wm**0.828) \
                  * (1. + b1 * wb**b2)

        def R_of_z(z):
            return 31.5 * wb * th**-4 * (z / 1.e3)**-1

        Rd, Req = R_of_z(self.zd), R_of_z(self.zeq)
        self.s = 2. / (3. * self.keq) * np.sqrt(6. / Req) * np.log(
            (np.sqrt(1. + Rd) + np.sqrt(Rd + Req)) / (1. + np.sqrt(Req)))
        self.ksilk = 1.6 * wb**0.52 * wm**0.73 \
                     * (1. + (10.4 * wm)**-0.95)               # 1/Mpc

        # ── CDM coefficients ──────────────────────────────────────────────────
        a1 = (46.9 * wm)**0.670 * (1. + (32.1 * wm)**-0.532)
        a2 = (12.0 * wm)**0.424 * (1. + (45.0 * wm)**-0.582)
        self.alpha_c = a1**(-wb / wm) * a2**(-(wb / wm)**3)
        bb1 = 0.944 / (1. + (458. * wm)**-0.708)
        bb2 = (0.395 * wm)**-0.0266
        self.beta_c = 1. / (1. + bb1 * (((wm - wb) / wm)**bb2 - 1.))

        # ── baryon coefficients ───────────────────────────────────────────────
        y = (1. + self.zeq) / (1. + self.zd)
        Gy = y * (-6. * np.sqrt(1. + y)
                  + (2. + 3. * y) * np.log(
                      (np.sqrt(1. + y) + 1.) / (np.sqrt(1. + y) - 1.)))
        self.alpha_b = 2.07 * self.keq * self.s * (1. + Rd)**-0.75 * Gy
        self.beta_b = 0.5 + wb / wm \
            + (3. - 2. * wb / wm) * np.sqrt((17.2 * wm)**2 + 1.)
        self.beta_node = 8.41 * wm**0.435

        # ── no-wiggle coefficients (eqs. 29-31) ──────────────────────────────
        self.a_gamma = 1. - 0.328 * np.log(431. * wm) * wb / wm \
                       + 0.38 * np.log(22.3 * wm) * (wb / wm)**2

    # T0-tilde, eq. 19-20
    def _T0(self, k, ac, bc):
        q = k / (13.41 * self.keq)
        C = 14.2 / ac + 386. / (1. + 69.9 * q**1.08)
        L = np.log(np.e + 1.8 * bc * q)
        return L / (L + C * q * q)

    def T_wiggle(self, k_hMpc):
        """Full EH98 transfer function; k in h/Mpc."""
        k = np.asarray(k_hMpc, dtype=np.float64) * self.h    # -> 1/Mpc
        ks = k * self.s

        # CDM
        f = 1. / (1. + (ks / 5.4)**4)
        Tc = f * self._T0(k, 1., self.beta_c) \
            + (1. - f) * self._T0(k, self.alpha_c, self.beta_c)

        # baryons
        stilde = self.s / (1. + (self.beta_node / ks)**3)**(1. / 3.)
        x = k * stilde
        j0 = np.where(x > 1.e-8, np.sin(x) / np.where(x == 0, 1., x),
                      1. - x * x / 6.)
        Tb = (self._T0(k, 1., 1.) / (1. + (ks / 5.2)**2)
              + self.alpha_b / (1. + (self.beta_b / ks)**3)
              * np.exp(-(k / self.ksilk)**1.4)) * j0

        wb, wm = self.wb, self.wm
        return (wb / wm) * Tb + ((wm - wb) / wm) * Tc

    def T_nowiggle(self, k_hMpc):
        """EH98 zero-baryon-oscillation (smooth) transfer; k in h/Mpc."""
        k = np.asarray(k_hMpc, dtype=np.float64) * self.h    # -> 1/Mpc
        gamma_eff = self.Om0 * self.h * (
            self.a_gamma + (1. - self.a_gamma) / (1. + (0.43 * k * self.s)**4))
        q = (k / self.h) * self.th**2 / gamma_eff       # k[h/Mpc] * Th^2 / Geff
        L0 = np.log(2. * np.e + 1.8 * q)
        C0 = 14.2 + 731. / (1. + 62.5 * q)
        return L0 / (L0 + C0 * q * q)

    def plin(self, k_hMpc, wiggle=True):
        """Unnormalized linear P(k) (amplitude absorbed into the bias fit)."""
        T = self.T_wiggle(k_hMpc) if wiggle else self.T_nowiggle(k_hMpc)
        return np.asarray(k_hMpc)**self.ns * T * T


# ═════════════════════════════════════════════════════════════════════════════
# 2. Spherical Bessel functions j0, j2, j4 (numpy-only, series at small x)
# ═════════════════════════════════════════════════════════════════════════════

def _sph_jn(n, x):
    x = np.asarray(x, dtype=np.float64)
    small = x < 0.35
    xs = np.where(small, 1., x)     # avoid division warnings
    s, c = np.sin(xs), np.cos(xs)
    if n == 0:
        big = s / xs
    elif n == 2:
        big = (3. / xs**3 - 1. / xs) * s - 3. / xs**2 * c
    elif n == 4:
        big = (105. / xs**5 - 45. / xs**3 + 1. / xs) * s \
              + (10. / xs**2 - 105. / xs**4) * c
    else:
        raise ValueError(n)
    # Taylor series: j_n(x) ~ x^n/(2n+1)!! * (1 - x^2/(2(2n+3)) + x^4/(8(2n+3)(2n+5)))
    dfact = {0: 1., 2: 15., 4: 945.}[n]
    x2 = x * x
    ser = x**n / dfact * (1. - x2 / (2. * (2 * n + 3))
                          + x2 * x2 / (8. * (2 * n + 3) * (2 * n + 5)))
    return np.where(small, ser, big)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Anisotropic BAO template: xi_ell(r) and xi(r_perp, r_par)
# ═════════════════════════════════════════════════════════════════════════════

class BAOTemplate:
    """
    Precomputes k-grids and P_lin / P_nw once; builds xi multipoles for a given
    (beta, Sig_par, Sig_perp) and evaluates the dilated 2D model.
    The overall amplitude bsq multiplies everything (profiled analytically).
    """

    def __init__(self, cosmo=None, kmin=1e-4, kmax=10., nk=1024,
                 rmin=1., rmax=300., nr=300, nmu=24, kdamp=0.5):
        # WARNING: the default EH98 fitting formula puts the sound horizon at
        # 101.77 Mpc/h, while CLASS gives 99.08 Mpc/h for AbacusSummit c000.
        # Fitting Abacus data with it biases alpha low by 2.7%.  Prefer
        #     from template_pk import class_pk;  BAOTemplate(cosmo=class_pk())
        # See --template class on the command line, and section 7's
        # cross-template selftest.
        self.cosmo = cosmo or EH98()
        self.k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.lnk = np.log(self.k)
        self.pw = self.cosmo.plin(self.k, wiggle=True)
        self.pnw = self.cosmo.plin(self.k, wiggle=False)
        # normalize to sigma8 (z=0) so bsq has a familiar magnitude; any
        # residual amplitude (growth, bias) is absorbed by the fitted bsq
        x = 8. * self.k
        W = 3. * (np.sin(x) - x * np.cos(x)) / x**3
        s8sq = _trapz(self.k**3 * self.pw / (2. * np.pi**2) * W * W,
                      np.log(self.k))
        norm = 0.8078**2 / s8sq          # AbacusSummit c000 sigma8
        self.pw *= norm
        self.pnw *= norm
        # apodization for the Bessel integrals (well below the BAO scale)
        self.apod = np.exp(-(self.k * kdamp)**2)
        self.r = np.linspace(rmin, rmax, nr)
        # Gauss-Legendre nodes on mu in [0, 1] (even integrand)
        x, w = np.polynomial.legendre.leggauss(nmu)
        self.mu = 0.5 * (x + 1.)
        self.wmu = 0.5 * w
        # cache Bessel kernels j_ell(k r): shape (nr, nk); ~10 MB each
        kr = self.r[:, None] * self.k[None, :]
        self._j = {0: _sph_jn(0, kr), 2: _sph_jn(2, kr), 4: _sph_jn(4, kr)}
        self._leg = {0: lambda m: np.ones_like(m),
                     2: lambda m: 0.5 * (3. * m**2 - 1.),
                     4: lambda m: 0.125 * (35. * m**4 - 30. * m**2 + 3.)}
        self._sign = {0: 1., 2: -1., 4: 1.}     # i^ell for even ell

    def xi_multipoles(self, beta, sig_par, sig_per):
        """Return dict ell -> xi_ell(self.r) for unit amplitude (bsq = 1)."""
        # memoize: Nelder-Mead revisits nearby (beta, Sig) many times
        key = (round(float(beta), 5), round(float(sig_par), 5),
               round(float(sig_per), 5))
        cache = getattr(self, '_xi_cache', None)
        if cache is None:
            cache = self._xi_cache = {}
        if key in cache:
            return cache[key]
        k, mu = self.k[None, :], self.mu[:, None]
        damp = np.exp(-0.5 * k * k * (mu * mu * sig_par**2
                                      + (1. - mu * mu) * sig_per**2))
        pkmu = (1. + beta * mu * mu)**2 * ((self.pw - self.pnw)[None, :] * damp
                                           + self.pnw[None, :])
        xi = {}
        for ell in (0, 2, 4):
            leg = self._leg[ell](self.mu)[:, None]
            # factor 2 from mu in [0,1]; (2l+1)/2 from the multipole projection
            pell = (2 * ell + 1) * np.sum(self.wmu[:, None] * leg * pkmu, axis=0)
            integ = self.k**3 * pell * self.apod          # k^2 dk = k^3 dlnk
            xi[ell] = self._sign[ell] / (2. * np.pi**2) \
                * _trapz(self._j[ell] * integ[None, :], self.lnk, axis=1)
        if len(cache) > 4096:
            cache.clear()
        cache[key] = xi
        return xi

    def xi2d(self, rt, rl, ap, at, xi_ell):
        """
        Dilated 2D correlation function on a grid.
        rt : (Nt,) transverse bin centres;  rl : (Nl,) LOS bin centres.
        Returns array (Nt, Nl).
        """
        rt_f = at * rt[:, None]
        rl_f = ap * rl[None, :]
        s = np.sqrt(rt_f**2 + rl_f**2)
        mu = np.where(s > 0, rl_f / np.where(s == 0, 1., s), 0.)
        out = np.zeros_like(s)
        for ell in (0, 2, 4):
            out += np.interp(s, self.r, xi_ell[ell],
                             left=xi_ell[ell][0], right=0.) \
                   * self._leg[ell](mu)
        return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. Nelder-Mead (hand-rolled, no scipy)
# ═════════════════════════════════════════════════════════════════════════════

def nelder_mead(func, x0, step=None, maxiter=4000, xtol=1e-6, ftol=1e-8):
    n = len(x0)
    x0 = np.asarray(x0, dtype=np.float64)
    step = np.asarray(step if step is not None else 0.05 * np.maximum(np.abs(x0), 0.1))
    simplex = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    fvals = [func(x) for x in simplex]
    for _ in range(maxiter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if (np.max([np.max(np.abs(s - simplex[0])) for s in simplex[1:]]) < xtol
                and abs(fvals[-1] - fvals[0]) < ftol):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])            # reflection
        fr = func(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            xe = centroid + 2. * (centroid - simplex[-1])   # expansion
            fe = func(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)  # contraction
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:                                           # shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


# ═════════════════════════════════════════════════════════════════════════════
# 5. The fit: chi2 with analytically profiled amplitude + broadband
# ═════════════════════════════════════════════════════════════════════════════

class BAOFitter:
    """
    Nonlinear params: theta = (ap, at, beta, sig_par, sig_per).
    Linear params (profiled exactly per chi2 call): overall amplitude bsq and
    smooth broadband  BB(s, mu) = sum_{i=0..2} sum_{l=0,2} c_il s^{-i} L_l(mu).
    """

    PRIORS = dict(ap=(0.8, 1.2), at=(0.8, 1.2), beta=(0.2, 4.0),
                  sig_par=(0.0, 15.0), sig_per=(0.0, 15.0))

    def __init__(self, template, rt, rl, rmin=30., rmax=180., use_hexa=True):
        self.tpl = template
        self.rt, self.rl = rt, rl
        RT, RL = np.meshgrid(rt, rl, indexing='ij')
        self.S = np.sqrt(RT**2 + RL**2)
        self.MU = np.where(self.S > 0, RL / np.where(self.S == 0, 1., self.S), 0.)
        self.mask = (self.S.ravel() > rmin) & (self.S.ravel() < rmax)
        # broadband design matrix on the masked bins
        s, m = self.S.ravel()[self.mask], self.MU.ravel()[self.mask]
        cols = []
        for i in (0, 1, 2):
            for leg in (np.ones_like(m), 0.5 * (3 * m**2 - 1)):
                cols.append(s**(-float(i)) * leg)
        self.BB = np.vstack(cols).T          # (nbin, 6)

    def _chi2_profiled(self, theta, data, ivar):
        ap, at, beta, sp, st = theta
        # soft prior walls
        pen = 0.
        for val, (lo, hi) in zip(theta, self.PRIORS.values()):
            if val < lo:
                pen += 1e4 * (lo - val)**2
            if val > hi:
                pen += 1e4 * (val - hi)**2
        ap = np.clip(ap, *self.PRIORS['ap'])
        at = np.clip(at, *self.PRIORS['at'])
        beta = np.clip(beta, *self.PRIORS['beta'])
        sp = np.clip(sp, *self.PRIORS['sig_par'])
        st = np.clip(st, *self.PRIORS['sig_per'])

        xi_ell = self.tpl.xi_multipoles(beta, sp, st)
        peak = self.tpl.xi2d(self.rt, self.rl, ap, at, xi_ell).ravel()[self.mask]
        # design matrix: [template, broadband...]; solve GLS for linear coeffs
        A = np.hstack([peak[:, None], self.BB])
        sw = np.sqrt(ivar)
        Aw = A * sw[:, None]
        try:
            coef = np.linalg.lstsq(Aw, data * sw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 1e30
        resid = data - A @ coef
        return float(np.sum(resid * resid * ivar)) + pen, coef

    def _profile_chi2(self, best, fixed_index, fixed_value, data, ivar):
        """Minimize chi2 over every nonlinear parameter not held fixed."""
        free = [i for i in range(5) if i != fixed_index]

        def objective(values):
            theta = np.array(best, copy=True)
            theta[fixed_index] = fixed_value
            theta[free] = values
            out = self._chi2_profiled(theta, data, ivar)
            return out if np.isscalar(out) else out[0]

        start = np.asarray(best)[free]
        step = np.array([0.005, 0.005, 0.1, 0.5, 0.5])[free]
        _, chi2 = nelder_mead(objective, start, step=step, maxiter=500)
        return float(chi2)

    def _profile_interval(self, best, chi2_min, index, data, ivar,
                          initial_step=0.004):
        """Profile-likelihood roots at Delta chi2 = 1."""
        lo_bound, hi_bound = list(self.PRIORS.values())[index]
        cache = {}

        def delta(value):
            value = float(value)
            if value not in cache:
                cache[value] = (self._profile_chi2(best, index, value,
                                                   data, ivar)
                                - chi2_min - 1.0)
            return cache[value]

        def bisect(a, b):
            fa, fb = delta(a), delta(b)
            if fa * fb > 0:
                return np.nan
            for _ in range(32):
                mid = 0.5 * (a + b)
                fm = delta(mid)
                if abs(fm) < 1e-5 or abs(b - a) < 1e-7:
                    return mid
                if fa * fm <= 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
            return 0.5 * (a + b)

        center = float(best[index])
        roots = []
        for direction, bound in ((-1.0, lo_bound), (1.0, hi_bound)):
            inner = center
            step = initial_step
            while direction * (bound - inner) > 0:
                outer = float(np.clip(inner + direction * step,
                                      lo_bound, hi_bound))
                if delta(outer) >= 0.0:
                    roots.append(bisect(min(center, outer),
                                        max(center, outer)))
                    break
                if outer == bound:
                    roots.append(np.nan)
                    break
                inner = outer
                step *= 1.5
            else:
                roots.append(np.nan)
        return roots[0], roots[1]

    def fit(self, xi2d_data, var2d, x0=None, verbose=False,
            compute_errors=True):
        data = xi2d_data.ravel()[self.mask]
        ivar = 1. / np.maximum(var2d.ravel()[self.mask], 1e-30)

        def f(theta):
            out = self._chi2_profiled(theta, data, ivar)
            return out if np.isscalar(out) else out[0]

        x0 = x0 or [1.0, 1.0, 1.5, 6.0, 3.0]
        best, chi2 = nelder_mead(f, x0, step=[0.02, 0.02, 0.3, 1.5, 1.0])
        # polish from the first solution
        best, chi2 = nelder_mead(f, best, step=[0.005, 0.005, 0.1, 0.5, 0.5])
        _, coef = self._chi2_profiled(best, data, ivar)
        res = dict(ap=best[0], at=best[1], beta=best[2],
                   sig_par=best[3], sig_per=best[4],
                   bsq=coef[0], bb=coef[1:], chi2=chi2,
                   ndof=int(self.mask.sum()) - 5 - len(coef))

        if compute_errors:
            for j, name in ((0, 'ap'), (1, 'at')):
                lower, upper = self._profile_interval(best, chi2, j,
                                                       data, ivar)
                err_lo = best[j] - lower if np.isfinite(lower) else np.nan
                err_hi = upper - best[j] if np.isfinite(upper) else np.nan
                res['sigma_' + name + '_lo'] = float(err_lo)
                res['sigma_' + name + '_hi'] = float(err_hi)
                res['sigma_' + name] = float(0.5 * (err_lo + err_hi))
        else:
            res['sigma_ap'] = res['sigma_at'] = np.nan
        if verbose:
            print("   ", {k: (round(v, 4) if np.isscalar(v) else v)
                          for k, v in res.items() if k != 'bb'})
        return res


# ═════════════════════════════════════════════════════════════════════════════
# 6. Data handling
# ═════════════════════════════════════════════════════════════════════════════

def load_xirppi(fn):
    d = np.load(fn)
    rp_bins, pi_bins, xi = d['rp_bins'], d['pi_bins'], d['xirppi']
    # Legacy compute_xirppi_from_xi3d returns a tuple that np.savez stores as
    # a (3, nrp, npi) array: (mean rp, mean pi, xi).  The final plane is the
    # correlation.  It is not a stack of raw and control-variate estimates.
    if xi.ndim == 3 and xi.shape[0] == 3:
        xi = xi[2]
    rt = 0.5 * (rp_bins[1:] + rp_bins[:-1])   # transverse
    rl = 0.5 * (pi_bins[1:] + pi_bins[:-1])   # line of sight
    return rt, rl, np.asarray(xi, dtype=np.float64)


def analyze(files, fitter_kwargs, label, template, xis=None, rt=None, rl=None):
    """Fit each realization + the mean; return summary dict."""
    if xis is None:
        rt0 = rl0 = None
        xis = []
        for fn in files:
            rt, rl, xi = load_xirppi(fn)
            if rt0 is None:
                rt0, rl0 = rt, rl
            assert np.allclose(rt, rt0) and np.allclose(rl, rl0)
            xis.append(xi)
        xis = np.array(xis)
        rt, rl = rt0, rl0
    nreal = len(xis)

    # diagonal variance of a single realization from the scatter across them
    if nreal >= 2:
        var1 = np.var(xis, axis=0, ddof=1)
        var1 = np.maximum(var1, 1e-4 * np.median(var1[var1 > 0]))
    else:
        print("WARNING: single realization -> uniform weights, errors from "
              "Delta chi^2 only")
        var1 = np.ones_like(xis[0])

    fitter = BAOFitter(template, rt, rl, **fitter_kwargs)

    print(f"\n=== {label}: {nreal} realization(s) ===")
    per = []
    for i in range(nreal):
        res = fitter.fit(xis[i], var1, compute_errors=False)
        per.append(res)
        print(f"  real {i}: ap = {res['ap']:.4f}  at = {res['at']:.4f}  "
              f"chi2/dof = {res['chi2']:.1f}/{res['ndof']}")

    res_mean = fitter.fit(np.mean(xis, axis=0), var1 / max(nreal, 1))
    aps = np.array([r['ap'] for r in per])
    ats = np.array([r['at'] for r in per])

    out = dict(label=label, aps=aps, ats=ats, mean_fit=res_mean, nreal=nreal)
    if nreal >= 2:
        out['ap_scatter'] = np.std(aps, ddof=1)
        out['at_scatter'] = np.std(ats, ddof=1)
        out['ap_err_mean'] = out['ap_scatter'] / np.sqrt(nreal)
        out['at_err_mean'] = out['at_scatter'] / np.sqrt(nreal)
    print(f"  mean fit: ap = {res_mean['ap']:.4f} +- {res_mean['sigma_ap']:.4f} "
          f"(dchi2=1) | at = {res_mean['at']:.4f} +- {res_mean['sigma_at']:.4f}")
    if nreal >= 2:
        print(f"  scatter over realizations: sigma(ap) = {out['ap_scatter']:.4f}"
              f"  sigma(at) = {out['at_scatter']:.4f}")
    return out


def report_improvement(pre, post):
    print("\n" + "=" * 64)
    print("   RECONSTRUCTION IMPROVEMENT (sigma_pre / sigma_post)")
    print("=" * 64)
    for key, name in (('sigma_ap', 'alpha_par '), ('sigma_at', 'alpha_perp')):
        a, b = pre['mean_fit'][key], post['mean_fit'][key]
        print(f"  {name}  (dchi2=1 on mean):    {a:.4f} -> {b:.4f}   "
              f"improvement x{a / b:.2f}")
    if 'ap_scatter' in pre and 'ap_scatter' in post:
        for key, name in (('ap_scatter', 'alpha_par '),
                          ('at_scatter', 'alpha_perp')):
            a, b = pre[key], post[key]
            print(f"  {name}  (realization scatter): {a:.4f} -> {b:.4f}   "
                  f"improvement x{a / b:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Self-test on synthetic data
# ═════════════════════════════════════════════════════════════════════════════

def _rs_of(tpl):
    """Sound horizon of a template's cosmology, in Mpc/h."""
    c = tpl.cosmo
    return float(getattr(c, "rs_drag", None) or c.s * c.h)


def get_template(name, z=2.5, **kw):
    """Build a BAOTemplate from a named linear P(k)."""
    if name == "eh98":
        return BAOTemplate(**kw)
    if name == "class":
        from template_pk import class_pk
        return BAOTemplate(cosmo=class_pk(z=z), **kw)
    raise ValueError(f"unknown template {name!r}")


def selftest(seed=42, gen="eh98", fit="eh98"):
    """
    Generate 6 'pre-recon' mocks (large damping) and 6 'post-recon' mocks
    (small damping, ap/at shifted) from the `gen` template + noise + a
    broadband contaminant, then fit them with the `fit` template.

    With gen == fit this only exercises the machinery -- it CANNOT detect a
    wrong sound horizon, because the same ruler is used on both sides.  That
    is precisely how the EH98 error survived the original selftest.  Run

        --selftest --gen class --fit eh98

    to see the failure: the recovered alpha comes back low by rs_EH98/rs_CLASS
    - 1 = 2.7%, and the test reports FAILED.
    """
    rng = np.random.default_rng(seed)
    print(f"Building templates: generate with {gen.upper()}, "
          f"fit with {fit.upper()}")
    tpl_gen = get_template(gen)
    tpl = tpl_gen if gen == fit else get_template(fit)
    edges = np.linspace(0., 200., 51)
    rt = rl = 0.5 * (edges[1:] + edges[:-1])

    def make_mocks(ap, at, beta, sp, st, bsq, nreal, noise):
        xi_ell = tpl_gen.xi_multipoles(beta, sp, st)
        base = bsq * tpl_gen.xi2d(rt, rl, ap, at, xi_ell)
        S = np.sqrt(rt[:, None]**2 + rl[None, :]**2)
        base = base + 2e-4 / np.maximum(S, 4.)**1.5     # broadband contaminant
        sig = noise * (1. + (60. / np.maximum(S, 4.))**2)
        return np.array([base + rng.normal(0., sig) for _ in range(nreal)])

    truth = dict(pre=dict(ap=1.015, at=0.985, sp=6.5, st=3.2),
                 post=dict(ap=1.015, at=0.985, sp=3.0, st=1.5))
    bsq, beta = 0.015, 1.6
    noise = 1.5e-5
    xis_pre = make_mocks(truth['pre']['ap'], truth['pre']['at'], beta,
                         truth['pre']['sp'], truth['pre']['st'],
                         bsq, 6, noise)
    xis_post = make_mocks(truth['post']['ap'], truth['post']['at'], beta,
                          truth['post']['sp'], truth['post']['st'],
                          bsq, 6, noise)

    kw = dict(rmin=30., rmax=180.)
    pre = analyze(None, kw, "SELFTEST pre-recon (Sig=6.5/3.2)", tpl,
                  xis=xis_pre, rt=rt, rl=rl)
    post = analyze(None, kw, "SELFTEST post-recon (Sig=3.0/1.5)", tpl,
                   xis=xis_post, rt=rt, rl=rl)

    print("\nTruth: ap = 1.0150, at = 0.9850 for both sets")
    if gen != fit:
        rg, rf = _rs_of(tpl_gen), _rs_of(tpl)
        print(f"Sound horizons: generated with {rg:.3f} Mpc/h, "
              f"fitted with {rf:.3f} Mpc/h -> expect alpha low by "
              f"{(1 - rg / rf) * 100:+.2f}%, i.e. ap ~ {1.015 * rg / rf:.4f}, "
              f"at ~ {0.985 * rg / rf:.4f}")
    report_improvement(pre, post)

    # pass if truth recovered within max(3 sigma, 0.3%) on the mean fits
    # (the 0.3% floor covers grid-interpolation systematics far below any
    # realistic Lya BAO error budget)
    def close(fitres, name, truthval):
        tol = max(3. * fitres['sigma_' + name], 3e-3)
        return abs(fitres[name] - truthval) < tol
    ok = (close(pre['mean_fit'], 'ap', 1.015) and close(pre['mean_fit'], 'at', 0.985)
          and close(post['mean_fit'], 'ap', 1.015) and close(post['mean_fit'], 'at', 0.985))
    print("\nSELFTEST:", "PASSED" if ok else "FAILED",
          "(truth recovered within max(3 sigma, 0.003) on the mean fits)")
    return ok


# ═════════════════════════════════════════════════════════════════════════════
# 8. Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prerecon', nargs='+', default=None,
                   help='npz files (rp_bins, pi_bins, xirppi) pre-recon')
    p.add_argument('--postrecon', nargs='+', default=None,
                   help='npz files post-recon')
    p.add_argument('--rmin', type=float, default=30.)
    p.add_argument('--rmax', type=float, default=180.)
    p.add_argument('--template', default='class', choices=['class', 'eh98'],
                   help="linear P(k) for the template.  'class' (default) uses "
                        "the true AbacusSummit sound horizon; 'eh98' uses the "
                        "fitting formula, which is 2.7%% long and biases alpha "
                        "low by the same amount")
    p.add_argument('--z', type=float, default=2.5, help='template redshift')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--gen', default='eh98', choices=['class', 'eh98'],
                   help='selftest: template the mocks are generated from')
    p.add_argument('--fit', default='eh98', choices=['class', 'eh98'],
                   help='selftest: template the mocks are fitted with. Set '
                        '--gen class --fit eh98 to make the selftest fail on '
                        'the sound-horizon error it used to hide')
    p.add_argument('--save', default=None, help='npz output for the summary')
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(gen=args.gen, fit=args.fit) else 1)

    if not args.prerecon:
        p.error("provide --prerecon (and ideally --postrecon), or --selftest")

    tpl = get_template(args.template, z=args.z)
    print(f"template: {args.template}  r_s,drag = {_rs_of(tpl):.3f} Mpc/h")
    kw = dict(rmin=args.rmin, rmax=args.rmax)
    pre = analyze(args.prerecon, kw, "pre-recon", tpl)
    post = None
    if args.postrecon:
        post = analyze(args.postrecon, kw, "post-recon", tpl)
        report_improvement(pre, post)

    if args.save:
        out = dict(pre_aps=pre['aps'], pre_ats=pre['ats'])
        for k in ('ap', 'at', 'sigma_ap', 'sigma_at', 'chi2'):
            out['pre_mean_' + k] = pre['mean_fit'][k]
        if post is not None:
            out['post_aps'] = post['aps']
            out['post_ats'] = post['ats']
            for k in ('ap', 'at', 'sigma_ap', 'sigma_at', 'chi2'):
                out['post_mean_' + k] = post['mean_fit'][k]
        np.savez(args.save, **out)
        print(f"\nSaved summary -> {args.save}")
