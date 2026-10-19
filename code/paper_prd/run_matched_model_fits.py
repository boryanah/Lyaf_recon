"""Validate and fit the predeclared common-template sensitivity comparison."""
import argparse
import resource
import time
import numpy as np
from matched_model_fit import OUT, MatchedPkModel, xi_fitter
from ic_bao_template import load_template, OUT as TEMPLATE
from estimator_bins import FourierBins
from target_bandpass import PREVIOUS, pass_power, save_atomic


def main(space, validate=False):
    cpu = time.process_time(); output = {}
    if space == 'xi':
        model = xi_fitter()
        output['basis_rank'] = model.broadband_rank
        output['singular_values'] = model.broadband_singular_values
    else:
        from spline_bao_fit import ProjectedFitter
        rule = FourierBins.load(PREVIOUS/'fourier_bins_order6.npz')
        mask = (rule.mean_k > .02) & (rule.mean_k < .4) & (rule.count > 0)
        model = MatchedPkModel(load_template(), rule, mask)
        mean = np.mean([np.load(OUT/f'measure_LOS{los}_ph{p:03d}.npz')['pre_raw_pk']
                        for los in ['z', 'y'] for p in range(6)], axis=0)
        ivar = rule.count[mask]/(2*mean[mask]**2)
        output['mask'] = mask; output['ivar'] = ivar
    if validate:
        if space == 'xi':
            known = np.load(TEMPLATE/'ic_xi_validation.npz')
            tags = [k.removesuffix('_xi') for k in known.files if k.endswith('_xi')]
            for tag in tags:
                fit = model.fit(known[tag+'_xi'])
                error = 100*(np.array([fit['ap'], fit['at']])-known[tag+'_input'])
                print(tag, 'error percent', error, flush=True)
                assert np.max(abs(error)) < .005, (tag, fit)
                output[tag+'_error_percent'] = error
                output[tag+'_input'] = known[tag+'_input']
                output[tag+'_chi2'] = fit['chi2']
        else:
            # The previous exact-mode IC tests independently validated the
            # physical model. Here test new broadband profiling and filtering.
            for ap, at in [(1., 1.), (1.003, .997), (.997, 1.003)]:
                theta = np.array([ap, at, 1., 4.6, 2.8])
                rng = np.random.default_rng(1024)
                smooth = model.BB@rng.normal(0., 2., model.BB.shape[1])
                data = .15**2*model.peak(*theta)+smooth
                fit = ProjectedFitter(model, data, ivar).fit()
                error = 100*(np.array([fit['ap'], fit['at']])-theta[:2])
                assert np.max(abs(error)) < .001, fit
                tag = f'{ap:.3f}_{at:.3f}'
                output[tag+'_error_percent'] = error
                print(tag, 'error percent', error, flush=True)
        dest = OUT/f'validation_{space}.npz'
    else:
        assert (OUT/f'validation_{space}.npz').exists()
        for los in ['z', 'y']:
            for state in ['pre', 'post']:
                for treatment in ['raw', 'lcv']:
                    fits = []
                    for phase in range(6):
                        if space == 'xi':
                            from target_bandpass import OUT as COMMON
                            row = np.load(COMMON/f'common_target_LOS{los}_ph{phase:03d}.npz')
                            fit = model.fit(row[f'w0.35_b512_n512_{state}_{treatment}'])
                        else:
                            row = np.load(OUT/f'measure_LOS{los}_ph{phase:03d}.npz')
                            fit = ProjectedFitter(model, row[f'{state}_{treatment}_pk'][mask], ivar).fit()
                        fits.append(fit)
                    tag = f'LOS{los}_{state}_{treatment}'
                    for key in fits[0]:
                        output[tag+'_'+key] = np.array([f[key] for f in fits])
                    print(space, tag, 'at percent', 100*(output[tag+'_at']-1), flush=True)
                    save_atomic(OUT/f'fits_{space}.partial.npz', **output)
        dest = OUT/f'fits_{space}.npz'
    output.update(cpu_seconds=time.process_time()-cpu,
                  peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    save_atomic(dest, **output)
    print('COMPLETE', dest, output['cpu_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--space', choices=['pk', 'xi'], required=True)
    p.add_argument('--validate', action='store_true')
    a = p.parse_args(); main(a.space, a.validate)
