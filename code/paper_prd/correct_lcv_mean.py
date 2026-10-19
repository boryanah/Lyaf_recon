#!/usr/bin/env python3
"""Replace the added-back mean only; preserve fields, coefficients, weights.

The correction is deterministic given the coefficient: CV_new = CV_old +
q*(E_new-E_old). This tests the effect of the surrogate-mean mismatch without
confounding it with a new reconstruction, coefficient fit, or covariance.
"""
from pathlib import Path
import resource
import time
import numpy as np
from estimator_bins import FourierBins
from ic_reference_spectrum import spectrum
from referee_lcv_audit import source_path
from template_pk import class_pk
from recon_paint_field import FOREST_RESPONSE

OUT=Path('/pscratch/sd/b/boryanah/Lyaf_recon/estimator_validation_20260908')


def main():
    begin=time.monotonic();cpu=time.process_time()
    reference=spectrum();old=class_pk(z=2.5)
    rule=FourierBins.load(OUT/'fourier_bins_order8.npz')
    for model in range(1,5):
        bf,beta=FOREST_RESPONSE[model]
        theory=np.sum(rule.weight*reference.plin(rule.k)*bf**2*(1+beta*rule.mu**2)**2,axis=1).reshape(100,5)
        for los in 'zy':
            src=np.load(source_path(model,los));out=dict(theory=theory,model=model,los=np.array(los))
            for state in ['pre','post']:
                q=src[f'm{model}_{state}_ensemble_wedges_coefficient']
                difference=theory-src[f'm{model}_theory_bandpower']
                out[state+'_bandpower']=src[f'm{model}_{state}_ensemble_wedges_bandpower']+q*difference
            mask=(rule.mean_k>.1)&(rule.mean_k<.4)
            observed=src[f'm{model}_control_bandpower'].mean(0).ravel()
            out['control_over_exact_mean']=np.average(observed[mask]/theory.ravel()[mask],weights=rule.count[mask])
            out['control_over_old_mean']=np.average(observed[mask]/src[f'm{model}_theory_bandpower'].mean(0).ravel()[mask],weights=rule.count[mask])
            print('FOURIER',model,los,out['control_over_exact_mean'],out['control_over_old_mean'],flush=True)
            np.savez(OUT/f'icmean_pk_Model{model}_LOS{los}.npz',**out)

    # The same separable, exact bin transform used in matched_lcv_bins.py,
    # now acting only on a deterministic ensemble-mean difference.
    nmax=int(np.ceil(.5*2000/(2*np.pi)));modes=np.arange(-nmax,nmax+1);kf=2*np.pi/2000
    kperp2=(modes[:,None]*kf)**2+(modes[None,:]*kf)**2
    kernels={}
    for n in [512,1152]:
        with np.load(OUT/f'xi_exact_bin_kernels_n{n}.npz') as cache:
            kernels[n]={key:cache[key] for key in ['transverse','radial']}
    bf,beta=FOREST_RESPONSE[1]
    for los in 'zy':
        for phase in range(6):
            destination=OUT/f'icmean_xi_LOS{los}_ph{phase:03d}.npz'
            if destination.exists():
                with np.load(destination) as prior:
                    if 'control_mean' in prior:
                        print('XI complete, skipping',los,phase,flush=True);continue
            src=np.load(OUT/f'matched_xi_LOS{los}_ph{phase:03d}.npz')
            out={k:src[k] for k in src.files if k not in ['elapsed_seconds','cpu_seconds','peak_rss_gb']}
            correction={n:{s:np.zeros((50,50)) for s in ['pre','post']} for n in kernels}
            for iz in range(nmax+1):
                k=np.sqrt(kperp2+(iz*kf)**2);good=(k>0)&(k<.5)
                mu2=np.divide((iz*kf)**2,k*k,out=np.zeros_like(k),where=k>0)
                difference=np.zeros_like(k)
                difference[good]=(reference.plin(k[good])-old.plin(k[good]))*bf**2*(1+beta*mu2[good])**2/2000**3
                for state in ['pre','post']:
                    q=np.interp(k,src['coefficient_k'],src[state+'_coefficient'],left=src[state+'_coefficient'][0],right=0.)
                    for n,ker in kernels.items():
                        correction[n][state]+=(ker['transverse']@(q*difference).ravel())[:,None]*ker['radial'][iz][None,:]*(1 if iz==0 else 2)
            for n in kernels:
                for state in ['pre','post']:
                    out[f'n{n}_{state}_mean_correction']=correction[n][state]
                    out[f'n{n}_{state}_lcv']+=correction[n][state]
                    out[f'n{n}_{state}_correction']+=correction[n][state]
            out['control_mean']=np.array('generating IC spectrum times recorded normalization and applied growth squared')
            np.savez(destination,**out)
            print('XI',los,phase,flush=True)
    np.savez(OUT/'ic_mean_correction_resources.npz',elapsed_seconds=time.monotonic()-begin,
             cpu_seconds=time.process_time()-cpu,peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)


if __name__=='__main__':main()
