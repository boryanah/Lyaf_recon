#!/usr/bin/env python3
"""Refit all phases with common target filtering, bins, and control weights."""
import argparse
import fcntl
import resource
import time
import numpy as np
from fit_common_target import CommonTargetFitter
from target_bandpass import OUT,save_atomic


def main(los,measure,start,small_scale_nl='dnl_arinyo',linear_template='vega'):
    suffix='' if small_scale_nl=='dnl_arinyo' else '_nonl'
    if linear_template=='class':suffix+='_class'
    dest=OUT/f'fits_common_LOS{los}_b{measure}_w{start:.2f}{suffix}.npz'
    lock=open(dest.with_suffix('.lock'),'a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('already running',dest,flush=True);return
    if dest.exists():
        print('exists',dest,flush=True);return
    validation=np.load(OUT/'common_target_validation.npz')
    assert 'cpu_seconds' in validation
    begin=time.monotonic();cpu=time.process_time()
    fitter=CommonTargetFitter(measure,start,small_scale_nl,linear_template)
    out=dict(los=np.array(los),measurement_grid=measure,start=start,
             small_scale_nl=np.array(small_scale_nl),linear_template=np.array(linear_template))
    keys=[f'n{target}_{state}_{kind}' for target in [512,1152]
          for state in ['pre','post'] for kind in ['raw','lcv']]
    rows=[]
    for phase in range(6):
        path=OUT/f'common_target_LOS{los}_ph{phase:03d}.npz'
        # Background pipeline: fit completed phase products while the next
        # target FFT runs. Waiting consumes no CPU and creates no outputs.
        while not path.exists():time.sleep(5)
        row=np.load(path);rows.append(row)
        for key in keys:
            fit=fitter.fit(row[f'w{start:.2f}_b{measure}_'+key])
            for par,value in fit.items():out.setdefault(key+'_'+par,[]).append(value)
            print(los,measure,start,key,phase,fit['ap'],fit['at'],flush=True)
        # The partial suffix is never accepted as a completed result.
        save_atomic(dest.with_suffix('.partial.npz'),**out,completed_phases=phase+1)
    for target in [512,1152]:
        for state in ['pre','post']:
            for kind in ['raw','lcv']:
                key=f'n{target}_{state}_{kind}'
                fields=np.array([d[f'w{start:.2f}_b{measure}_'+key] for d in rows])
                fit=fitter.fit(fields.mean(0))
                for par in fit:out[key+'_'+par+'_stack']=fit[par]
    out.update(cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-begin,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    save_atomic(dest,**out);print('COMPLETE',dest,'CPU',out['cpu_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--los',choices=['y','z'],required=True)
    p.add_argument('--measure',type=int,choices=[512,1152],default=512)
    p.add_argument('--start',type=float,choices=[.35,.40],default=.35)
    p.add_argument('--small-scale-nl',choices=['dnl_arinyo','none'],default='dnl_arinyo')
    p.add_argument('--linear-template',choices=['vega','class'],default='vega')
    a=p.parse_args();main(a.los,a.measure,a.start,a.small_scale_nl,a.linear_template)
