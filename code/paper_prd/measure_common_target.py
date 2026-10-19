#!/usr/bin/env python3
"""Match output bandpass, control, and measurement lattice, without new fields."""
import argparse
import gc
import importlib.util
import resource
import time
from pathlib import Path
import numpy as np
from target_bandpass import (OUT,PREVIOUS,LBOX,KF,IMAX,MODES,compact_modes,
                              pass_power,input_taper,plane,add_plane,load_kernels,save_atomic)
from highres_two_orientation_xirppi import linear_modes,post_path
from recon_paint_field import OUT_DIR,FOREST_RESPONSE
from ic_reference_spectrum import spectrum


def main(phase,los,workers=4):
    dest=OUT/f'common_target_LOS{los}_ph{phase}.npz'
    if dest.exists():
        print('already exists',dest,flush=True);return
    begin=time.monotonic();cpu=time.process_time()
    prior=np.load(PREVIOUS/f'icmean_xi_LOS{los}_ph{phase}.npz')
    spec=importlib.util.spec_from_file_location('lyaf_abacus_asdf',
        '/global/u1/b/boryanah/.local/perlmutter/python-3.11/lib/python3.11/site-packages/abacusnbody/data/asdf.py')
    ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
    import asdf
    ext.set_nthreads(workers);asdf.get_config().add_extension(ext.AbacusExtension())
    print(phase,los,'IC modes',flush=True)
    delta=linear_modes(phase,los,512,float(prior['growth']),workers)
    linear=np.ascontiguousarray(delta[np.ix_(MODES%512,MODES%512,np.arange(IMAX+1))])
    del delta;gc.collect()
    sim=f'AbacusSummit_base_c000_ph{phase}'
    prepath=OUT_DIR/f'dF_ref_Model_1_LOS{los}_nmesh512_{sim}_z2.500.npy'
    print(phase,los,'reference and reconstructed target modes',flush=True)
    pre=compact_modes(prepath,'node',workers)
    low=compact_modes(post_path(phase,los,512,512,OUT_DIR),'continuous',workers)
    highpath=post_path(phase,los,1152,512,OUT_DIR/'two_orientation_1152')
    high=compact_modes(highpath,'continuous',workers)
    kernels={n:load_kernels(n) for n in [512,1152]}
    reference=spectrum();bf,beta=FOREST_RESPONSE[1]
    output=dict(phase=int(phase),los=np.array(los),coefficient_k=prior['coefficient_k'],
                starts=np.array([.35,.40]),stop=.5,
                source_pre=np.array(str(prepath)),source_post1152=np.array(str(highpath)),
                mean=np.array('actual IC generating spectrum and applied growth'),
                pre1152=np.array('same shipped modes, native taper ratio applied to node-corrected 512 reference'))
    for start in [.35,.40]:
        for measure in kernels:
            for target in [512,1152]:
                for state in ['pre','post']:
                    for kind in ['raw','lcv']:
                        output[f'w{start:.2f}_b{measure}_n{target}_{state}_{kind}']=np.zeros((50,50))
    # Actual paired target transfer, with no BAO template fit or fitted rescaling.
    sums={key:np.zeros(500) for key in ['low','high','cross','difference','count']}
    maximum_pre_difference=0.
    for iz in range(IMAX+1):
        k,mu2=plane(iz);good=(k>0)&(k<.5)
        mu=np.sqrt(mu2);ik=np.floor(k/.005).astype(int);im=np.minimum((mu*5).astype(int),4)
        index=(ik*5+im)[good];multiplicity=1 if iz==0 else 2
        products=dict(low=abs(low[:,:,iz])**2,high=abs(high[:,:,iz])**2,
                      cross=np.real(high[:,:,iz]*np.conj(low[:,:,iz])),
                      difference=abs(high[:,:,iz]-low[:,:,iz])**2,count=np.ones_like(k))
        for key,value in products.items():
            sums[key]+=np.bincount(index,weights=multiplicity*value[good],minlength=500)
        # Both pre fields contain exactly the same underlying shipped modes.
        # Their only non-roundoff difference here is the documented input taper.
        ratio_xy=(input_taper(MODES[:,None]*KF,1152)*input_taper(MODES[None,:]*KF,1152)/
                  (input_taper(MODES[:,None]*KF,512)*input_taper(MODES[None,:]*KF,512)))
        ratio=ratio_xy*input_taper(iz*KF,1152)/input_taper(iz*KF,512)
        if np.any(good):maximum_pre_difference=max(maximum_pre_difference,float(np.max(abs(ratio[good]-1))))
        response=bf*(1+beta*mu2)
        mean=np.zeros_like(k);mean[good]=reference.plin(k[good])*response[good]**2/LBOX**3
        control_residual=abs(linear[:,:,iz]*response)**2-mean
        for state in ['pre','post']:
            q=np.interp(k,prior['coefficient_k'],prior[state+'_coefficient'],
                        left=prior[state+'_coefficient'][0],right=0.)
            for target in [512,1152]:
                target_power=(abs(pre[:,:,iz])**2*(ratio**2 if target==1152 else 1.) if state=='pre'
                              else products['low' if target==512 else 'high'])
                for start in [.35,.40]:
                    window=np.where(good,pass_power(k,start),0.)
                    for kind in ['raw','lcv']:
                        power=window*(target_power if kind=='raw' else target_power-q*control_residual)
                        for measure,ker in kernels.items():
                            add_plane(output[f'w{start:.2f}_b{measure}_n{target}_{state}_{kind}'],power,ker,iz)
    for key,value in sums.items():output['post_transfer_'+key]=value.reshape(100,5)
    output.update(max_pre_amplitude_difference=maximum_pre_difference,
                  elapsed_seconds=time.monotonic()-begin,cpu_seconds=time.process_time()-cpu,
                  peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    save_atomic(dest,**output)
    print('COMPLETE',dest,'CPU',output['cpu_seconds'],'RSS GiB',output['peak_rss_gb'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True)
    p.add_argument('--los',choices=['y','z'],required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();main(f'{int(a.phase):03d}',a.los,a.workers)
