#!/usr/bin/env python3
"""Regenerate the Model 1 LCV stability scan with the correct IC mean."""
from pathlib import Path
import resource
import time
import numpy as np
from estimator_bins import FourierBins
from fit_binned_pk import BinnedPkModel,PkmuFitter,class_pk,source_path,bin_mask

HERE=Path(__file__).resolve().parent
OUT=Path('/pscratch/sd/b/boryanah/Lyaf_recon/estimator_validation_20260908')
PARS=['ap','at','beta','sig_par','sig_per','chi2']
for los in 'zy':
    begin=time.monotonic();cpu=time.process_time();out={}
    src=np.load(source_path(1,los));audit=np.load(HERE/'data'/f'referee_lcv_Model1_LOS{los}.npz')
    corrected=np.load(OUT/f'icmean_pk_Model1_LOS{los}.npz')
    rule=FourierBins.load(OUT/'fourier_bins_order6.npz');cosmo=class_pk(z=2.5)
    for kmax,nbb in [(.30,3),(.35,3),(.40,2),(.40,3),(.40,4)]:
        mask=bin_mask(src['k'],src['m1_pre_raw_bandpower'].mean(0),src['N'],.02,kmax).ravel()
        model=BinnedPkModel(cosmo,rule,mask,nbb,False)
        for kind in ['raw','cv']:
            for state in ['pre','post']:
                fields=src[f'm1_{state}_raw_bandpower'] if kind=='raw' else corrected[state+'_bandpower']
                results=[]
                for phase in range(6):
                    source_kind='raw' if kind=='raw' else 'ensemble_wedges'
                    theta=[src[f'm1_{state}_{source_kind}_kmax{kmax:.2f}_nbb{nbb}_{p}_phase'][phase] for p in PARS[:5]]
                    fit=PkmuFitter(model,fields[phase].ravel()[mask],1/audit[f'{state}_{kind}_variance'].ravel()[mask]).fit(x0=theta,compute_errors=False)
                    results.append(fit)
                for par in PARS:out[f'{kind}_{state}_kmax{kmax:.2f}_nbb{nbb}_{par}']=np.array([r[par] for r in results])
            print(los,kind,kmax,nbb,flush=True)
        np.savez(OUT/f'icmean_stability_Model1_LOS{los}.npz',**out)
    out.update(elapsed_seconds=time.monotonic()-begin,cpu_seconds=time.process_time()-cpu,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    np.savez(OUT/f'icmean_stability_Model1_LOS{los}.npz',**out)
