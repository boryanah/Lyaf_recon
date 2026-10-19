#!/usr/bin/env python3
"""Paired-phase summary of matched target and painting tests."""
import numpy as np
from target_bandpass import OUT,PREVIOUS


def fmt(values):
    values=np.asarray(values)
    return f'{values.mean():+.6f} +/- {values.std(ddof=1)/np.sqrt(len(values)):.6f}'


def load_pair(measure,start,suffix=''):
    paths=[OUT/f'fits_common_LOS{los}_b{measure}_w{start:.2f}{suffix}.npz' for los in 'zy']
    if not all(p.exists() for p in paths):return None
    return [np.load(p) for p in paths]


def main():
    print('All offsets/responses in percent; six within-phase orientation averages.')
    for measure,start,suffix in [(512,.35,''),(1152,.35,''),(512,.40,''),
                                  (512,.35,'_nonl'),(512,.35,'_nonl_class')]:
        rows=load_pair(measure,start,suffix)
        if rows is None:continue
        print('\nCOMMON BANDPASS',measure,start,suffix)
        for kind in ['raw','lcv']:
            for par in ['ap','at']:
                deltas=[]
                for target in [512,1152]:
                    pre=np.mean([d[f'n{target}_pre_{kind}_{par}'] for d in rows],axis=0)
                    post=np.mean([d[f'n{target}_post_{kind}_{par}'] for d in rows],axis=0)
                    delta=100*(post-pre);deltas.append(delta)
                    print(target,kind,par,'pre',fmt(100*(pre-1)),'post',fmt(100*(post-1)),
                          'paired',fmt(delta))
                print('TARGET HIGH MINUS LOW',kind,par,fmt(deltas[1]-deltas[0]))
        print('LCV post alpha_perp by orientation; phase averages are used above, not twelve independent boxes.')
        for los,d in zip('zy',rows):
            print(los,fmt(100*(d['n1152_post_lcv_at']-1)),
                  'paired',fmt(100*(d['n1152_post_lcv_at']-d['n1152_pre_lcv_at'])))
        for target in [512,1152]:
            for state in ['pre','post']:
                for kind in ['raw','lcv']:
                    key=f'n{target}_{state}_{kind}_'
                    limits=np.concatenate([d[key+'at_limit'] for d in rows])
                    if limits.any():
                        damping=np.array([[d[key+p] for p in ['sigmaNL_par','sigmaNL_per']] for d in rows])
                        print('BOUNDARY',key,'flags',int(limits.sum()),
                              'damping min',damping.min(axis=(0,2)),
                              'damping max',damping.max(axis=(0,2)))
    low=load_pair(512,.35);high=load_pair(1152,.35)
    if low is not None and high is not None:
        print('\nCHANGE ONLY THE MEASUREMENT LATTICE, 1152 minus 512')
        for target in [512,1152]:
            for kind in ['raw','lcv']:
                for par in ['ap','at']:
                    delta=[np.mean([100*(d[f'n{target}_post_{kind}_{par}']-
                                       d[f'n{target}_pre_{kind}_{par}']) for d in rows],axis=0)
                           for rows in [low,high]]
                    print(target,kind,par,fmt(delta[1]-delta[0]))
    translation=OUT/'paint_translation_validation.npz'
    if translation.exists():
        d=np.load(translation);print('\nKNOWN TRANSLATION: numerical offsets, no physical acoustic motion')
        for n in [512,1152]:
            for damping in ['prelike','postlike']:
                for name in ['exact','zero','typical','halfcell','phase_average']:
                    print(n,damping,name,*[100*(float(d[f'n{n}_{damping}_{name}_{p}'])-1) for p in ['ap','at']])
    paths=sorted(OUT.glob('common_target_LOS*_ph???.npz'))
    if len(paths)==12:
        print('\nDIRECT POST-FIELD TRANSFER; five mu bins; no fitted acoustic template.')
        sums={key:np.zeros((100,5)) for key in ['low','high','cross','difference']}
        for path in paths:
            with np.load(path) as d:
                for key in sums:sums[key]+=d['post_transfer_'+key]
        for i in [10,30,50,70,90]:
            s={key:value[i-2:i+3].sum(0) for key,value in sums.items()}
            print('k',(i+.5)*.005,'power high/low minus one [%]',100*(s['high']/s['low']-1),
                  'rms complex-mode difference [%]',100*np.sqrt(s['difference']/s['low']))
    cpu=peak=0.
    for path in OUT.glob('*.npz'):
        if '.partial.' in path.name:continue
        with np.load(path) as d:
            if 'cpu_seconds' in d:cpu+=float(d['cpu_seconds'])
            if 'peak_rss_gb' in d:peak=max(peak,float(d['peak_rss_gb']))
    print('\nRECORDED CPU HOURS',cpu/3600,'PEAK RSS GiB',peak)


if __name__=='__main__':main()
