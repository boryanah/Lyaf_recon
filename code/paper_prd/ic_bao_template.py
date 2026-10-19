"""A full/smooth BAO template made from the actual IC generating spectrum."""
from pathlib import Path
import resource
import time
import numpy as np
from ic_reference_spectrum import spectrum, CACHE
from template_pk import TabulatedPk, class_pk
from target_bandpass import save_atomic

OUT = Path('/pscratch/sd/b/boryanah/Lyaf_recon/template_closure_20260909')


def load_template(engine='savgol'):
    path = OUT / f'ic_bao_{engine}.npz'
    with np.load(path) as d:
        return TabulatedPk(d['k'], d['full'], d['smooth'], float(d['rs_drag']),
                           label=f'actual IC/{engine}')


def main():
    from cosmoprimo import PowerSpectrumInterpolator1D, PowerSpectrumBAOFilter
    begin = time.process_time()
    reference = spectrum()
    with np.load(CACHE) as d:
        k = d['k']; full = d['power_surrogate']
    # Keep the original full spectrum, including its table/interpolation.
    # The separately computed CLASS spectrum supplies only the reported rd,
    # not the full or smooth power used in this template.
    interp = PowerSpectrumInterpolator1D(k, full, extrap_kmin=1e-7, extrap_kmax=100.)
    for engine in ['savgol', 'hinton2017']:
        filt = PowerSpectrumBAOFilter(interp, engine=engine)
        smooth = np.asarray(filt.smooth_pk_interpolator()(k))
        assert np.isfinite(smooth).all() and np.all(smooth > 0)
        save_atomic(OUT / f'ic_bao_{engine}.npz', k=k, full=full, smooth=smooth,
                    rs_drag=class_pk().rs_drag, source=np.array(str(CACHE)),
                    engine=np.array(engine))
        test = np.geomspace(.002, .6, 1200)
        np.testing.assert_array_equal(load_template(engine).plin(test), reference.plin(test))
        print(engine, 'exact full-spectrum interpolation preserved', flush=True)
    save_atomic(OUT/'template_build_resources.npz', cpu_seconds=time.process_time()-begin,
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)


if __name__ == '__main__':
    main()
