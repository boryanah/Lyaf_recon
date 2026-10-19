"""Report every new fit comparison with six phase-paired LOS uncertainties."""
from pathlib import Path
import numpy as np
from ic_bao_template import OUT


def stat(values, reference=0.):
    phase = 100*(np.asarray(values).mean(axis=0)-reference)
    return f'{phase.mean():+.6f} +/- {phase.std(ddof=1)/np.sqrt(6):.6f}'


def main():
    print('# Template closure results: all values in percent')
    print('Mean LOSy/LOSz within each phase; six phase standard errors.\n')
    for model in range(1,5):
        d = [dict(np.load(OUT/f'spline_Model{model}_LOS{los}.npz')) for los in ['y','z']]
        extra = [dict(np.load(OUT/f'spline_extra_Model{model}_LOS{los}.npz')) for los in ['y','z']]
        print(f'## Model {model}\n')
        print('| Template / treatment / weight | Pre parallel | Post parallel | Delta parallel | Pre transverse | Post transverse | Delta transverse |')
        print('| --- | --- | --- | --- | --- | --- | --- |')
        for case in ['poly3_ic','spline06_class','spline06_ic','spline05_ic','spline08_ic','hinton2017','savgol']:
            for treatment, weight in [('raw','raw'),('cv','raw'),('cv','cv'),('raw','cv')]:
                if case == 'savgol' and (treatment,weight) != ('raw','cv'):continue
                source = extra if case in ['hinton2017','savgol'] else d
                if case != 'savgol' and (treatment,weight) == ('raw','cv'):continue
                fields=[]
                for par in ['ap','at']:
                    v=[np.array([row[f'{case}_{treatment}_{weight}_{s}_{par}'] for row in source]) for s in ['pre','post']]
                    fields.extend([stat(v[0],1),stat(v[1],1),stat(v[1]-v[0])])
                print('| '+' / '.join([case,treatment,weight])+' | '+' | '.join(fields)+' |')
        print()
    print('## Exact-IC Model 1 configuration space\n')
    print('| Band / correction / treatment | Pre parallel | Post parallel | Delta parallel | Pre transverse | Post transverse | Delta transverse |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for band in ['native','common']:
        for nl in ['none','dnl_arinyo']:
            ds=[dict(np.load(OUT/f'ic_xi_LOS{los}_{band}_{nl}.npz')) for los in ['y','z']]
            for kind in ['raw','lcv']:
                fields=[]
                for par in ['ap','at']:
                    v=[np.array([d[f'{kind}_{s}_{par}'] for d in ds]) for s in ['pre','post']]
                    fields.extend([stat(v[0],1),stat(v[1],1),stat(v[1]-v[0])])
                print('| '+' / '.join([band,nl,kind])+' | '+' | '.join(fields)+' |')
    print('\n## Validation and resources\n')
    validation=np.load(OUT/'spline_validation.npz')
    for model in ['poly3','spline05','spline06','spline08']:
        for shape in ['none','saturation','curvature']:
            keys=[k[:-6] for k in validation if k.startswith(model+'_') and k.endswith('_input') and '_'+shape+'_' in k]
            error=np.array([100*(validation[k+'_recovered']-validation[k+'_input'])[:2] for k in keys])
            retain=np.array([validation[k+'_retention'] for k in keys])
            print(model,shape,'max absolute alpha errors',abs(error).max(0),'derivative norm retained',retain.min(0),retain.max(0))
    cpu=0.;rss=0.;count=0;bounds=0;alpha_bounds=0;min_bsq=np.inf;start=0.
    for file in sorted(OUT.glob('*.npz')):
        if '.partial.' in file.name:continue
        d=np.load(file)
        if 'total_cpu_seconds' in d:cpu+=float(d['total_cpu_seconds'])
        elif 'cpu_seconds' in d and d['cpu_seconds'].ndim==0:cpu+=float(d['cpu_seconds'])
        if 'peak_rss_gb' in d:rss=max(rss,float(d['peak_rss_gb']))
        if not file.name.startswith(('spline_Model','spline_extra','ic_xi_LOS')):continue
        for key in d:
            if key.endswith('_valid'):
                assert np.all(d[key]);count+=d[key].size
            if key.endswith('_near_bounds'):
                bounds+=np.any(d[key],axis=-1).sum();alpha_bounds+=np.any(d[key][:,:2],axis=-1).sum()
            if key.endswith('_bsq'):min_bsq=min(min_bsq,d[key].min())
            if key.endswith('_start_disagreement'):start=max(start,d[key].max())
    print('Simulation fits',count,'Fourier any near boundary',bounds,'alpha near boundary',alpha_bounds)
    print('Minimum Fourier amplitude',min_bsq,'max alternative-start alpha difference percent',100*start)
    print('Recorded CPU seconds',cpu,'hours',cpu/3600,'max worker RSS GiB',rss)


if __name__=='__main__':main()
