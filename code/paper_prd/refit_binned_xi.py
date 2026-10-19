#!/usr/bin/env python3
"""Bin-averaged raw, original-control, and matched-control xi refits."""
import argparse
import fcntl
from pathlib import Path
import resource
import time
import numpy as np
from fit_binned_xi import BinnedXiFitter,PARS

OUT=Path('/pscratch/sd/b/boryanah/Lyaf_recon/estimator_validation_20260908')
DATA=Path(__file__).resolve().parent/'data'
HIGH=Path('/pscratch/sd/b/boryanah/Lyaf_recon/two_orientation_1152/xi')


def run(nmesh,los,order=6,small_scale_nl='dnl_arinyo',kinds=None,linear_template='vega',control_mean='original'):
    begin=time.monotonic();cpu=time.process_time()
    suffix='' if small_scale_nl=='dnl_arinyo' else '_nonl'
    if linear_template=='class':suffix+='_class'
    if control_mean=='ic':suffix+='_icmean'
    dest=OUT/f'binned_xi_n{nmesh}_LOS{los}_o{order}{suffix}.npz'
    # Permit independent orientation screens without duplicate writes when
    # a sequential wrapper subsequently reaches the same product.
    lock=open(dest.with_suffix('.lock'),'a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('already running',dest,flush=True);return
    if dest.exists():
        with np.load(dest) as previous:
            if 'cpu_seconds' in previous:
                print('already complete',dest,flush=True);return
    validation=np.load(OUT/'xi_dilation_validation.npz');assert 'elapsed_seconds' in validation
    prefix='matched' if control_mean=='original' else 'icmean'
    rows=[np.load(OUT/f'{prefix}_xi_LOS{los}_ph{p:03d}.npz') for p in range(6)]
    fits=BinnedXiFitter(nmesh,order=order,small_scale_nl=small_scale_nl,linear_template=linear_template)
    out=dict(nmesh=nmesh,los=np.array(los),order=order,phases=np.arange(6),
             rmin=30.,rmax=180.,quadrature_points=len(fits.rule.rp),bin_count=fits.rule.nbins,
             small_scale_nl=np.array(small_scale_nl),linear_template=np.array(linear_template),
             control_mean=np.array(control_mean))
    for kind in (kinds or ['raw','lcv','original_lcv']):
        for state in ['pre','post']:
            if kind=='original_lcv':
                if nmesh==1152:
                    fields=np.array([np.load(HIGH/f'xi_highres_Model1_LOS{los}_ph{p:03d}.npz')[f'{state}_lcv'] for p in range(6)])
                else:
                    suffix='' if los=='z' else '_LOSy'
                    fields=np.array([np.load(DATA/f'xi_lcv_{state}_Model1{suffix}_ph{p:03d}.npz')['xirppi_lcv'] for p in range(6)])
            else:fields=np.array([d[f'n{nmesh}_{state}_{kind}'] for d in rows])
            results=[]
            for phase,field in enumerate(fields):
                result=fits.fit(field);results.append(result)
                print(nmesh,los,kind,state,phase,[(p,result[p]) for p in ['ap','at']],flush=True)
            for key in results[0]:out[f'{kind}_{state}_{key}']=np.array([r[key] for r in results])
            stack=fits.fit(fields.mean(0))
            for key in stack:out[f'{kind}_{state}_{key}_stack']=stack[key]
            np.savez(dest,**out)
        for par in ['ap','at']:
            delta=100*(out[f'{kind}_post_{par}']-out[f'{kind}_pre_{par}'])
            print('SUMMARY',nmesh,los,kind,par,delta.mean(),delta.std(ddof=1)/np.sqrt(6),flush=True)
    out.update(elapsed_seconds=time.monotonic()-begin,cpu_seconds=time.process_time()-cpu,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    np.savez(dest,**out);print('COMPLETE',dest,out['elapsed_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nmesh',type=int,required=True);p.add_argument('--los',choices=['y','z'],required=True)
    p.add_argument('--order',type=int,default=6)
    p.add_argument('--small-scale-nl',choices=['dnl_arinyo','none'],default='dnl_arinyo')
    p.add_argument('--kinds',nargs='+',choices=['raw','lcv','original_lcv'])
    p.add_argument('--linear-template',choices=['vega','class'],default='vega')
    p.add_argument('--control-mean',choices=['original','ic'],default='original')
    a=p.parse_args();run(a.nmesh,a.los,a.order,a.small_scale_nl,a.kinds,a.linear_template,a.control_mean)
