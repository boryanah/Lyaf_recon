"""All-model fixed-weight template and spline tests on archived spectra."""
import argparse
import fcntl
from pathlib import Path
import resource
import time
import numpy as np
from estimator_bins import FourierBins
from fit_binned_pk import BinnedPkModel, class_pk, source_path, bin_mask
from spline_bao_fit import SplinePkModel, ProjectedFitter, PARS
from ic_bao_template import OUT, load_template
from target_bandpass import PREVIOUS, save_atomic

CASES = [('poly3_ic', None, 'ic'), ('spline06_class', .06, 'class'),
         ('spline06_ic', .06, 'ic'), ('spline05_ic', .05, 'ic'),
         ('spline08_ic', .08, 'ic')]


def main(model_no, los):
    dest = OUT/f'spline_Model{model_no}_LOS{los}.npz'
    lock = open(dest.with_suffix('.lock'), 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('already running', dest, flush=True); return
    if dest.exists():
        print('exists', dest, flush=True); return
    with np.load(OUT/'spline_validation.npz') as validation:
        assert 'cpu_seconds' in validation
    cpu = time.process_time()
    src = np.load(source_path(model_no, los))
    audit = np.load(Path(__file__).resolve().parent/'data'/f'referee_lcv_Model{model_no}_LOS{los}.npz')
    corrected = np.load(PREVIOUS/f'icmean_pk_Model{model_no}_LOS{los}.npz')
    previous = np.load(PREVIOUS/f'binned_pk_Model{model_no}_LOS{los}_o6_icmean.npz')
    rule = FourierBins.load(PREVIOUS/'fourier_bins_order6.npz')
    mask = bin_mask(src['k'], src[f'm{model_no}_pre_raw_bandpower'].mean(0), src['N'], .02, .4).ravel()
    np.testing.assert_array_equal(rule.count, src['N'].ravel())
    out = dict(model=model_no, los=np.array(los), mask=mask)
    for label, spacing, waveform in CASES:
        cosmo = load_template() if waveform == 'ic' else class_pk()
        model = (BinnedPkModel(cosmo, rule, mask, 3, True) if spacing is None
                 else SplinePkModel(cosmo, rule, mask, spacing))
        for treatment, weight in [('raw', 'raw'), ('cv', 'raw'), ('cv', 'cv')]:
            for state in ['pre', 'post']:
                spectra = (src[f'm{model_no}_{state}_raw_bandpower'] if treatment == 'raw'
                           else corrected[state+'_bandpower']).reshape(6, -1)[:, mask]
                ivar = 1/audit[f'pre_{weight}_variance'].ravel()[mask]
                results = []
                for phase in range(6):
                    start = [previous[f'peak3_{treatment}_{weight}_{state}_{p}'][phase] for p in PARS]
                    fit = ProjectedFitter(model, spectra[phase], ivar).fit(start)
                    results.append(fit)
                tag = f'{label}_{treatment}_{weight}_{state}'
                for key in results[0]:
                    out[tag+'_'+key] = np.array([r[key] for r in results])
                print(model_no, los, tag, 'at offsets [%]',
                      100*(out[tag+'_at']-1), flush=True)
                save_atomic(dest.with_suffix('.partial.npz'), **out)
    out.update(cpu_seconds=time.process_time()-cpu,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    save_atomic(dest, **out)
    print('COMPLETE', dest, out['cpu_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=int, choices=[1,2,3,4], required=True)
    p.add_argument('--los', choices=['y','z'], required=True)
    a = p.parse_args(); main(a.model, a.los)
