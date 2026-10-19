"""
template_pk.py
--------------
A drop-in replacement for the EH98 class inside fit_bao_apat_standalone.py,
backed by a real Boltzmann P(k).

Why this exists
---------------
The EH98 fitting formula gets the sound horizon wrong by 2.7 % for a Planck
cosmology: it gives r_s,drag = 101.77 Mpc/h where CLASS gives 99.08 Mpc/h for
AbacusSummit c000.  Fitting AbacusSummit data with an EH98 template therefore
returns alpha ~ 0.974 instead of 1.000, and

    0.9742 x (101.77 / 99.08) = 1.0008

accounts for the whole shift.  The fitter's machinery is fine; only the
template's ruler was wrong.

The class exposes exactly the interface BAOTemplate expects --
`plin(k_hMpc, wiggle=True|False)` -- so it can be passed straight in.

The wiggle/no-wiggle split uses cosmoprimo's PowerSpectrumBAOFilter rather than
the EH98 zero-baryon formula, so the smooth component is a genuine de-wiggling
of the same P(k) instead of a different cosmology's broadband.

The default filter is `savgol`.  A good de-wiggling must satisfy P_w/P_nw -> 1
outside the oscillatory range; measured on this cosmology at z=2.5,

    engine         |P/P_nw-1|, k<0.005    |P/P_nw-1|, 2<k<20
    savgol               0.00003               0.00037
    hinton2017           0.00537               0.00227
    wallish2018          0.02382               0.00000
    ehpoly               0.02746               0.00000

wallish2018 and ehpoly leave a ~2.5% non-oscillatory excess at low k, which
then rides along inside the (P_w - P_nw) "peak" component of the template.
savgol is the only one clean at both ends.

Caching
-------
CLASS takes ~30 s to run, so the tabulated (k, P_w, P_nw) is written to an npz.
After the first call this module needs only numpy, which keeps
fit_bao_apat_standalone.py runnable in a bare environment.

Usage
-----
  python template_pk.py --z 2.5 --build      # build/refresh the cache, plot it
  >>> from template_pk import class_pk
  >>> cosmo = class_pk(z=2.5)
  >>> tpl = BAOTemplate(cosmo=cosmo)
"""

import argparse
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / "templates"
KMIN, KMAX, NK = 1e-5, 100.0, 4096


class TabulatedPk:
    """Log-log interpolated P(k) with a matched no-wiggle companion.

    Mirrors the EH98 interface used by BAOTemplate: `plin(k, wiggle=)`.
    `rs_drag` is carried along because it is the actual ruler being fitted --
    quote it whenever you quote an alpha.
    """

    def __init__(self, k, pw, pnw, rs_drag, sigma8=None, label="class"):
        self.k = np.asarray(k, dtype=np.float64)
        self._lnk = np.log(self.k)
        self._lpw = np.log(np.asarray(pw, dtype=np.float64))
        self._lpnw = np.log(np.asarray(pnw, dtype=np.float64))
        self.rs_drag = float(rs_drag)          # Mpc/h
        self.sigma8 = sigma8
        self.label = label
        # log-log slopes for power-law extrapolation beyond the table
        self._slo = (self._lpw[1] - self._lpw[0]) / (self._lnk[1] - self._lnk[0])
        self._shi = (self._lpw[-1] - self._lpw[-2]) / (self._lnk[-1] - self._lnk[-2])

    def plin(self, k_hMpc, wiggle=True):
        lk = np.atleast_1d(np.log(np.asarray(k_hMpc, dtype=np.float64)))
        lp = self._lpw if wiggle else self._lpnw
        out = np.empty_like(lk)
        # power-law extrapolation on both ends (np.interp would clamp instead)
        lo = lk < self._lnk[0]
        hi = lk > self._lnk[-1]
        if lo.any():
            out[lo] = lp[0] + self._slo * (lk[lo] - self._lnk[0])
        if hi.any():
            out[hi] = lp[-1] + self._shi * (lk[hi] - self._lnk[-1])
        mid = ~(lo | hi)
        out[mid] = np.interp(lk[mid], self._lnk, lp)
        return np.exp(out)

    def __repr__(self):
        return (f"TabulatedPk({self.label}, rs_drag={self.rs_drag:.3f} Mpc/h, "
                f"k in [{self.k[0]:.1e}, {self.k[-1]:.1e}])")


def _cache_path(z, cosmology, engine):
    return CACHE_DIR / f"pk_{cosmology}_{engine}_z{z:.3f}.npz"


def build_cache(z=2.5, cosmology="AbacusSummit", filter_engine="savgol",
                verbose=True):
    """Run CLASS via cosmoprimo, de-wiggle, and write the npz cache."""
    from cosmoprimo.fiducial import AbacusSummit, DESI
    from cosmoprimo import PowerSpectrumBAOFilter

    cos = {"AbacusSummit": AbacusSummit, "DESI": DESI}[cosmology](engine="class")
    if verbose:
        print(f"  running CLASS ({cosmology}, z={z}) ...", flush=True)

    pk = cos.get_fourier().pk_interpolator(of="delta_m").to_1d(z=z)
    k = np.logspace(np.log10(KMIN), np.log10(KMAX), NK)
    pw = pk(k)

    bao = PowerSpectrumBAOFilter(pk, engine=filter_engine, cosmo=cos)
    pnw = bao.smooth_pk_interpolator()(k)

    rs = float(cos.rs_drag)            # cosmoprimo returns Mpc/h
    s8 = float(cos.sigma8_m)
    if verbose:
        print(f"  rs_drag = {rs:.4f} Mpc/h   sigma8(z=0) = {s8:.4f}")
        # sanity: the wiggle/no-wiggle ratio should oscillate about 1 with a
        # few-percent amplitude and be featureless outside the BAO range
        m = (k > 0.02) & (k < 0.5)
        print(f"  |P/P_nw - 1| max in 0.02<k<0.5 = "
              f"{np.abs(pw[m] / pnw[m] - 1).max():.4f}")
        m2 = k < 5e-3
        print(f"  |P/P_nw - 1| max at k<0.005    = "
              f"{np.abs(pw[m2] / pnw[m2] - 1).max():.4f}  (should be << 0.01)")

    os.makedirs(CACHE_DIR, exist_ok=True)
    fn = _cache_path(z, cosmology, filter_engine)
    np.savez(fn, k=k, pw=pw, pnw=pnw, rs_drag=rs, sigma8=s8, z=z)
    if verbose:
        print(f"  cached -> {fn}")
    return fn


def class_pk(z=2.5, cosmology="AbacusSummit", filter_engine="savgol",
             rebuild=False, verbose=False):
    """Return a TabulatedPk, building the CLASS cache on first use."""
    fn = _cache_path(z, cosmology, filter_engine)
    if rebuild or not fn.exists():
        build_cache(z=z, cosmology=cosmology, filter_engine=filter_engine,
                    verbose=True)
    d = np.load(fn)
    tp = TabulatedPk(d["k"], d["pw"], d["pnw"], float(d["rs_drag"]),
                     sigma8=float(d["sigma8"]),
                     label=f"{cosmology}/CLASS z={float(d['z']):.2f}")
    if verbose:
        print(f"  {tp}")
    return tp


def eh98_rs(**kw):
    """The EH98 sound horizon, in Mpc/h, for comparison."""
    from fit_bao_apat_standalone import EH98
    e = EH98(**kw)
    return e.s * e.h          # EH98.s is in Mpc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z", type=float, default=2.5)
    p.add_argument("--cosmology", default="AbacusSummit")
    p.add_argument("--filter", dest="filter_engine", default="savgol")
    p.add_argument("--build", action="store_true", help="force a rebuild")
    p.add_argument("--plot", default=None, help="path for a comparison figure")
    a = p.parse_args()

    tp = class_pk(z=a.z, cosmology=a.cosmology,
                  filter_engine=a.filter_engine, rebuild=a.build, verbose=True)

    rs_eh = eh98_rs()
    print(f"\n  r_s,drag   CLASS : {tp.rs_drag:8.4f} Mpc/h")
    print(f"  r_s,drag   EH98  : {rs_eh:8.4f} Mpc/h")
    print(f"  ratio EH98/CLASS : {rs_eh / tp.rs_drag:8.5f}   "
          f"-> alpha bias of {(rs_eh / tp.rs_drag - 1) * 100:.2f}% "
          f"if EH98 is used on AbacusSummit data")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from fit_bao_apat_standalone import EH98

        e = EH98()
        k = np.logspace(-3, 0, 800)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        # match amplitudes at k=0.02 so only the shapes are compared
        pc, pe = tp.plin(k), e.plin(k)
        pe = pe * np.interp(0.02, k, pc) / np.interp(0.02, k, pe)
        ax[0].loglog(k, pc, lw=1.6, label=f"CLASS ({a.cosmology})")
        ax[0].loglog(k, pe, lw=1.2, ls="--", label="EH98")
        ax[0].set_xlabel("k [h/Mpc]"); ax[0].set_ylabel("P(k)")
        ax[0].legend(); ax[0].set_title("linear P(k)")

        ax[1].semilogx(k, tp.plin(k) / tp.plin(k, wiggle=False) - 1, lw=1.6,
                       label=f"CLASS / no-wiggle ({a.filter_engine})")
        ax[1].semilogx(k, e.plin(k) / e.plin(k, wiggle=False) - 1, lw=1.2,
                       ls="--", label="EH98 / EH98-nw")
        ax[1].axhline(0, color="k", lw=.6)
        ax[1].set_xlim(0.01, 0.6); ax[1].set_xlabel("k [h/Mpc]")
        ax[1].set_ylabel(r"$P/P_{\rm nw}-1$")
        ax[1].set_title(f"BAO wiggles -- note the phase drift "
                        f"({rs_eh/tp.rs_drag:.4f} in $r_s$)")
        ax[1].legend()
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print(f"  figure -> {a.plot}")
