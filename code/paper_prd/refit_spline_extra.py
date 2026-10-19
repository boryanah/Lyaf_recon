"""Complete fixed-weight raw/LCV comparison and a de-wiggling sensitivity."""
import argparse
import resource
import time
from pathlib import Path
import numpy as np
from estimator_bins import FourierBins, exact_fourier_modes
from fit_binned_pk import source_path, bin_mask, exact_template
from spline_bao_fit import SplinePkModel, ProjectedFitter, PARS
from ic_bao_template import OUT, load_template
from target_bandpass import PREVIOUS, save_atomic


def main(model_no, los):
    dest = OUT/f'spline_extra_Model{model_no}_LOS{los}.npz'
    if dest.exists():
        print('exists', dest, flush=True); return
    cpu = time.process_time(); out = {}
    src = np.load(source_path(model_no, los))
    audit = np.load(Path(__file__).resolve().parent/'data'/f'referee_lcv_Model{model_no}_LOS{los}.npz')
    corrected = np.load(PREVIOUS/f'icmean_pk_Model{model_no}_LOS{los}.npz')
    primary = np.load(OUT/f'spline_Model{model_no}_LOS{los}.npz')
    rule = FourierBins.load(PREVIOUS/'fourier_bins_order6.npz')
    mask = bin_mask(src['k'], src[f'm{model_no}_pre_raw_bandpower'].mean(0), src['N'], .02, .4).ravel()
    for engine in ['savgol', 'hinton2017']:
        model = SplinePkModel(load_template(engine), rule, mask, .06)
        cases = [('raw','cv')] if engine == 'savgol' else [('raw','raw'),('cv','raw'),('cv','cv')]
        if engine == 'hinton2017':
            rule.exact = exact_fourier_modes()
            for state, sp, st in [('pre',4.6,2.8),('post',2.2,1.4)]:
                for ap, at in [(1.,1.),(1.003,.997),(.997,1.003)]:
                    truth = np.array([ap,at,2.1,sp,st])
                    data = .15**2*exact_template(rule,truth,load_template(engine),True)[mask]
                    ivar = 1/audit['pre_cv_variance'].ravel()[mask]
                    fit = ProjectedFitter(model,data,ivar).fit(multistart=False)
                    errors = 100*(np.array([fit['ap'],fit['at']])-truth[:2])
                    out[f'closure_{state}_{ap:.3f}_{at:.3f}'] = errors
                    if np.max(abs(errors)) > .003:
                        raise RuntimeError(f'Hinton closure failed: {errors}')
        for treatment, weight in cases:
            for state in ['pre','post']:
                spectra = (src[f'm{model_no}_{state}_raw_bandpower'] if treatment == 'raw'
                           else corrected[state+'_bandpower']).reshape(6,-1)[:,mask]
                ivar = 1/audit[f'pre_{weight}_variance'].ravel()[mask]
                tag = f'{engine}_{treatment}_{weight}_{state}'
                # The same saved starting convention, not a unity constraint.
                start_tag = f'spline06_ic_{treatment}_{weight if treatment == "cv" else "raw"}_{state}'
                fits = [ProjectedFitter(model,row,ivar).fit(
                    [primary[start_tag+'_'+p][i] for p in PARS]) for i,row in enumerate(spectra)]
                for key in fits[0]:
                    out[tag+'_'+key] = np.array([fit[key] for fit in fits])
                print(model_no,los,tag,100*(out[tag+'_at']-1),flush=True)
    out.update(cpu_seconds=time.process_time()-cpu,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    save_atomic(dest, **out)
    print('COMPLETE',dest,out['cpu_seconds'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=int,choices=[1,2,3,4],required=True)
    p.add_argument('--los',choices=['y','z'],required=True)
    a=p.parse_args();main(a.model,a.los)
