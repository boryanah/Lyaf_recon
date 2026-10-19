"""Exact-IC template test on archived native and common-band Model 1 xi."""
import argparse
import fcntl
import resource
import time
import numpy as np
from fit_binned_xi import BinnedXiFitter
from fit_common_target import CommonTargetFitter
from ic_bao_template import OUT
from target_bandpass import OUT as BAND, PREVIOUS, save_atomic


def main(los, band, small_scale_nl):
    dest = OUT/f'ic_xi_LOS{los}_{band}_{small_scale_nl}.npz'
    lock = open(dest.with_suffix('.lock'), 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('already running', dest, flush=True); return
    if dest.exists():
        print('exists', dest, flush=True); return
    with np.load(OUT/'ic_xi_validation.npz') as d:
        assert 'cpu_seconds' in d
    cpu = time.process_time()
    if band == 'native':
        fitter = BinnedXiFitter(1152, small_scale_nl=small_scale_nl,
                               linear_template='ic_savgol')
        rows = [np.load(PREVIOUS/f'icmean_xi_LOS{los}_ph{p:03d}.npz') for p in range(6)]
        prefix = ''
    else:
        fitter = CommonTargetFitter(512, .35, small_scale_nl=small_scale_nl,
                                   linear_template='ic_savgol')
        rows = [np.load(BAND/f'common_target_LOS{los}_ph{p:03d}.npz') for p in range(6)]
        prefix = 'w0.35_b512_'
    out = dict(los=np.array(los), band=np.array(band),
               small_scale_nl=np.array(small_scale_nl))
    for kind in ['raw', 'lcv']:
        for state in ['pre', 'post']:
            tag = f'{kind}_{state}'
            fits = [fitter.fit(row[prefix+f'n1152_{state}_{kind}']) for row in rows]
            for key in fits[0]:
                out[tag+'_'+key] = np.array([fit[key] for fit in fits])
            print(los, band, small_scale_nl, tag, 'offsets [%]',
                  100*(out[tag+'_at']-1), flush=True)
            save_atomic(dest.with_suffix('.partial.npz'), **out)
    out.update(cpu_seconds=time.process_time()-cpu,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    save_atomic(dest, **out)
    print('COMPLETE', dest, out['cpu_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--los', choices=['y', 'z'], required=True)
    p.add_argument('--band', choices=['native', 'common'], required=True)
    p.add_argument('--small-scale-nl', choices=['none', 'dnl_arinyo'], default='none')
    a = p.parse_args(); main(a.los, a.band, a.small_scale_nl)
