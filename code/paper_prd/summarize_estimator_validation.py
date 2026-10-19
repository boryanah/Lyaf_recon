#!/usr/bin/env python3
"""Print paired-phase summaries; never count LOSy/z as twelve boxes."""
from pathlib import Path
import argparse
import numpy as np
from scipy.stats import t

OUT=Path('/pscratch/sd/b/boryanah/Lyaf_recon/estimator_validation_20260908')


def stats(a):
    a=np.asarray(a);return a.mean(),a.std(ddof=1)/np.sqrt(len(a))


def fmt(a):
    m,s=stats(a);return f'{m:+.5f} +/- {s:.5f}'


def load_xi(nmesh,los,suffix):
    """Raw refits are unchanged; corrected files intentionally store only LCV."""
    with np.load(OUT/f'binned_xi_n{nmesh}_LOS{los}_o6.npz') as d:
        result={k:d[k] for k in d.files if k.startswith('raw_')}
    with np.load(OUT/f'binned_xi_n{nmesh}_LOS{los}_o6{suffix}.npz') as d:
        result.update({k:d[k] for k in d.files})
    return result


def main(control_mean='ic'):
    suffix='_icmean' if control_mean=='ic' else ''
    print('All offsets and responses below are in percent. Six phase averages.\n')
    for model in range(1,5):
        paths=[OUT/f'binned_pk_Model{model}_LOS{los}_o6{suffix}.npz' for los in 'zy']
        if not all(p.exists() for p in paths):continue
        rows=[np.load(p) for p in paths]
        print('FOURIER Model',model)
        for variant in ['full3','full4','peak3']:
            for treatment,weight in [('raw','raw'),('raw','cv'),('cv','raw'),('cv','cv'),('raw','state'),('cv','state')]:
                tag=f'{variant}_{treatment}_{weight}'
                if not all(tag+'_post_at' in r for r in rows):continue
                pre=np.mean([r[tag+'_pre_at'] for r in rows],axis=0)
                post=np.mean([r[tag+'_post_at'] for r in rows],axis=0)
                print(tag,'pre',fmt(100*(pre-1)),'post',fmt(100*(post-1)),
                      'delta',fmt(100*(post-pre)))
    for nmesh in [512,1152]:
        paths=[OUT/f'binned_xi_n{nmesh}_LOS{los}_o6{suffix}.npz' for los in 'zy']
        if not all(p.exists() for p in paths):continue
        rows=[load_xi(nmesh,los,suffix) for los in 'zy']
        print('\nCONFIGURATION nmesh',nmesh)
        for kind in ['raw','lcv','original_lcv']:
            for par in ['ap','at']:
                if not all(f'{kind}_post_{par}' in r for r in rows):continue
                pre=np.mean([r[f'{kind}_pre_{par}'] for r in rows],axis=0)
                post=np.mean([r[f'{kind}_post_{par}'] for r in rows],axis=0)
                delta=100*(post-pre);mean,sem=stats(delta)
                print(kind,par,'pre',fmt(100*(pre-1)),'post',fmt(100*(post-1)),
                      'delta',fmt(delta),'p',2*t.sf(abs(mean/sem),5))
                dz,dy=[100*(r[f'{kind}_post_{par}']-r[f'{kind}_pre_{par}']) for r in rows]
                print('  LOSz',fmt(dz),'LOSy',fmt(dy),'y-z',fmt(dy-dz),'corr',np.corrcoef(dz,dy)[0,1])
    paths=[OUT/f'binned_xi_n{n}_LOS{los}_o6{suffix}.npz' for n in [512,1152] for los in 'zy']
    if all(p.exists() for p in paths):
        rows=[load_xi(n,los,suffix) for n in [512,1152] for los in 'zy']
        print('\nHIGH MINUS LOW, paired response difference')
        for kind in ['raw','lcv','original_lcv']:
            for par in ['ap','at']:
                if not all(f'{kind}_post_{par}' in r for r in rows):continue
                delta=[100*(r[f'{kind}_post_{par}']-r[f'{kind}_pre_{par}']) for r in rows]
                print(kind,par,fmt((delta[2]+delta[3]-delta[0]-delta[1])/2))
    for variant in ['_nonl','_nonl_class']:
        paths=[OUT/f'binned_xi_n1152_LOS{los}_o6{variant}{suffix}.npz' for los in 'zy']
        if not all(p.exists() for p in paths):continue
        rows=[np.load(p) for p in paths]
        print('\nCONFIGURATION TEMPLATE SENSITIVITY',variant)
        for par in ['ap','at']:
            pre=np.mean([r[f'lcv_pre_{par}'] for r in rows],axis=0)
            post=np.mean([r[f'lcv_post_{par}'] for r in rows],axis=0)
            print(par,'pre',fmt(100*(pre-1)),'post',fmt(100*(post-1)),
                  'delta',fmt(100*(post-pre)))
    print('\nPERIODIC NOISE-FREE CLOSURE')
    for path in sorted(OUT.glob('periodic_xi_validation_n*.npz')):
        d=np.load(path);print(path.name)
        for state in ['pre','post']:
            if state+'_ap' in d:print(state,100*(d[state+'_ap']-1),100*(d[state+'_at']-1))
        if 'post_ap' in d:print('delta',100*(d['post_ap']-d['pre_ap']),100*(d['post_at']-d['pre_at']))
    total=0.;peak=0.
    for path in OUT.glob('*.npz'):
        with np.load(path) as d:
            if 'cpu_seconds' in d:total+=float(d['cpu_seconds'])
            if 'peak_rss_gb' in d:peak=max(peak,float(d['peak_rss_gb']))
    print('\nRecorded process CPU hours (excludes setup/failed attempts/validation lacking CPU metadata)',total/3600,
          '\nLargest measured peak RSS GiB',peak)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-mean',choices=['ic','original'],default='ic')
    main(parser.parse_args().control_mean)
