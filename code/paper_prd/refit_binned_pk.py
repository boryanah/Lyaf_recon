#!/usr/bin/env python3
"""Refit raw/controlled simulations with validated mode-bin averaging."""
import argparse
import fcntl
from pathlib import Path
import resource
import time
import numpy as np
from estimator_bins import FourierBins
from fit_binned_pk import BinnedPkModel,PkmuFitter,class_pk,source_path,bin_mask

OUT=Path('/pscratch/sd/b/boryanah/Lyaf_recon/estimator_validation_20260908')
HERE=Path(__file__).resolve().parent
PARS=('ap','at','beta','sig_par','sig_per','chi2','bsq')


def run(model,los,order=6,control_mean='original'):
    begin=time.monotonic();cpu=time.process_time()
    suffix='' if control_mean=='original' else '_icmean'
    dest=OUT/f'binned_pk_Model{model}_LOS{los}_o{order}{suffix}.npz'
    lock=open(dest.with_suffix('.lock'),'a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('already running',dest,flush=True);return
    corrected=np.load(OUT/f'icmean_pk_Model{model}_LOS{los}.npz') if suffix else None
    src=np.load(source_path(model,los));audit=np.load(HERE/'data'/f'referee_lcv_Model{model}_LOS{los}.npz')
    validation=np.load(OUT/'fourier_dilation_validation.npz')
    assert 'elapsed_seconds' in validation
    rule=FourierBins.load(OUT/f'fourier_bins_order{order}.npz')
    np.testing.assert_array_equal(rule.count,src['N'].ravel())
    mask=bin_mask(src['k'],src[f'm{model}_pre_raw_bandpower'].mean(0),src['N'],.02,.40).ravel()
    cosmo=class_pk(z=2.5)
    out=dict(model=model,los=np.array(los),order=order,phases=np.arange(6),
             source=np.array(str(source_path(model,los))),mask=mask,control_mean=np.array(control_mean))
    if suffix:
        # Raw spectra and all weights are unchanged. Reuse raw fits exactly,
        # without charging their old runtime to this new run.
        with np.load(OUT/f'binned_pk_Model{model}_LOS{los}_o{order}.npz') as base:
            out.update({k:base[k] for k in base.files if '_raw_raw_' in k or '_raw_cv_' in k or '_raw_state_' in k})
    previous_wall=previous_cpu=0.
    if dest.exists():
        with np.load(dest) as old:
            out.update({k:old[k] for k in old.files})
        previous_wall=float(out.pop('elapsed_seconds',0.))
        previous_cpu=float(out.pop('cpu_seconds',0.))
    for variant,nbb,peakonly in [('full3',3,False),('full4',4,False),('peak3',3,True)]:
        template=BinnedPkModel(cosmo,rule,mask,nbb,peakonly)
        cases=[('raw','raw'),('cv','raw'),('cv','cv')]
        if variant=='full3':cases += [('raw','cv'),('raw','state'),('cv','state')]
        if variant=='full4':cases=[('cv','raw'),('cv','cv')]
        for treatment,weight in cases:
            if all(f'{variant}_{treatment}_{weight}_{state}_{p}' in out
                   for state in ['pre','post'] for p in PARS):
                continue
            for state in ['pre','post']:
                kind='raw' if treatment=='raw' else 'ensemble_wedges'
                spectra=src[f'm{model}_{state}_{kind}_bandpower'].reshape(6,-1)[:,mask]
                if treatment=='cv' and corrected is not None:
                    spectra=corrected[state+'_bandpower'].reshape(6,-1)[:,mask]
                variance_key=f'{state}_{treatment}_variance' if weight=='state' else f'pre_{weight}_variance'
                variance=audit[variance_key].ravel()[mask]
                fits=[]
                for phase in range(6):
                    start=np.array([src[f'm{model}_{state}_{kind}_kmax0.40_nbb3_{p}_phase'][phase]
                                    for p in PARS[:5]])
                    fit=PkmuFitter(template,spectra[phase],1/variance).fit(x0=start,compute_errors=False)
                    if not all(np.isfinite(fit[p]) for p in PARS):raise RuntimeError('nonfinite fit')
                    fits.append(fit)
                tag=f'{variant}_{treatment}_{weight}_{state}'
                for par in PARS:out[f'{tag}_{par}']=np.array([f[par] for f in fits])
            delta=100*(out[f'{variant}_{treatment}_{weight}_post_at']-out[f'{variant}_{treatment}_{weight}_pre_at'])
            print(model,los,variant,treatment,weight,'delta_at',delta.mean(),delta.std(ddof=1)/np.sqrt(6),flush=True)
            np.savez(dest,**out)
    out.update(elapsed_seconds=previous_wall+time.monotonic()-begin,cpu_seconds=previous_cpu+time.process_time()-cpu,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    np.savez(dest,**out);print('COMPLETE',dest,out['elapsed_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=int,required=True)
    p.add_argument('--los',choices=['y','z'],required=True);p.add_argument('--order',type=int,default=6)
    p.add_argument('--control-mean',choices=['original','ic'],default='original')
    a=p.parse_args();run(a.model,a.los,a.order,a.control_mean)
