#!/usr/bin/env python3
"""One identical LCV correction evaluated on both target lattices.

Uses held-out high-resolution monopole coefficients, identical interpolation,
and a shared |k|<0.5 support. Exact separable bin kernels avoid constructing
1152^3 correlation grids. Only the phase IC FFT is large (<30 GB peak).
"""
import argparse
import gc
import importlib.util
from pathlib import Path
import resource
import time
import numpy as np
from scipy.fft import fft2,irfftn
from highres_two_orientation_xirppi import linear_modes,Geometry,bin_rppi,LBOX
from template_pk import class_pk
from recon_paint_field import FOREST_RESPONSE

OUT=Path('/pscratch/sd/b/boryanah/Lyaf_recon/estimator_validation_20260908')
HIGH=Path('/pscratch/sd/b/boryanah/Lyaf_recon/two_orientation_1152/xi')
DATA=Path(__file__).resolve().parent/'data'
KMAX=.5
IMAX=int(np.ceil(KMAX*LBOX/(2*np.pi)))
MODES=np.arange(-IMAX,IMAX+1)
KF=2*np.pi/LBOX


def kernels(nmesh):
    path=OUT/f'xi_exact_bin_kernels_n{nmesh}.npz'
    if path.exists():
        d=np.load(path);return d['transverse'],d['radial']
    ix=np.arange(nmesh);distance=np.minimum(ix,nmesh-ix)*LBOX/nmesh
    transverse_bin=np.floor(np.sqrt(distance[:,None]**2+distance[None,:]**2)/4).astype(int)
    radial_bin=np.floor(distance/4).astype(int)
    pick=MODES%nmesh
    transverse=[];radial=[]
    for b in range(50):
        mask=transverse_bin==b
        transform=fft2(mask.astype(np.float64),workers=1)/mask.sum()
        selected=transform[np.ix_(pick,pick)]
        assert np.max(abs(selected.imag))<1e-12
        transverse.append(selected.real.ravel())
        radial.append(np.cos(np.arange(IMAX+1)[:,None]*KF*distance[radial_bin==b]).mean(1))
    transverse=np.array(transverse);radial=np.array(radial).T
    zero=int(np.flatnonzero(MODES==0)[0])
    np.testing.assert_allclose(transverse[:,zero*len(MODES)+zero],1,atol=1e-14)
    np.testing.assert_allclose(radial[0],1,atol=1e-14)
    np.savez(path,transverse=transverse,radial=radial,nmesh=nmesh,kmax=KMAX,modes=MODES)
    return transverse,radial


def run(phase,los,workers=4):
    start=time.monotonic();cpu=time.process_time();dest=OUT/f'matched_xi_LOS{los}_ph{phase}.npz'
    if dest.exists():
        print('exists',dest,flush=True);return
    high=[np.load(HIGH/f'xi_highres_Model1_LOS{los}_ph{p:03d}.npz') for p in range(6)]
    growth=float(high[int(phase)]['growth_z99_to_z2p5'])
    q={s:np.mean([row[f'{s}_coefficient'] for p,row in enumerate(high) if p!=int(phase)],axis=0)
       for s in ['pre','post']}
    geo=Geometry(1152);cosmo=class_pk(z=2.5);bf,beta=FOREST_RESPONSE[1]
    both_kernels={n:kernels(n) for n in [512,1152]}
    print(phase,los,'loading and cropping IC',flush=True)
    # The numerical environment lacks the Abacus ASDF entry point. Load the
    # user's existing pure-Python extension without importing its old NumPy
    # environment or changing any installation.
    import asdf
    extension_path=Path('/global/u1/b/boryanah/.local/perlmutter/python-3.11/lib/python3.11/site-packages/abacusnbody/data/asdf.py')
    spec=importlib.util.spec_from_file_location('lyaf_abacus_asdf',extension_path)
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    extension.set_nthreads(workers)
    asdf.get_config().add_extension(extension.AbacusExtension())
    delta=linear_modes(phase,los,512,growth,workers)
    pick=MODES%512;zero=np.flatnonzero(MODES==0)[0]
    correction={n:{s:np.zeros((50,50)) for s in q} for n in both_kernels}
    # For phase 000 also verify this direct bin transform against a full
    # 512^3 inverse FFT using exactly the same correction power.
    check_power=np.zeros(delta.shape,dtype=np.float64) if phase=='000' else None
    kperp2=(MODES[:,None]*KF)**2+(MODES[None,:]*KF)**2
    for iz in range(IMAX+1):
        kval=np.sqrt(kperp2+(iz*KF)**2)
        good=(kval>0)&(kval<KMAX)
        mu2=np.divide((iz*KF)**2,kval**2,out=np.zeros_like(kval),where=kval>0)
        response=bf*(1+beta*mu2)
        linear=delta[:,:,iz][np.ix_(pick,pick)]*response
        theory=np.zeros_like(kval)
        theory[good]=cosmo.plin(kval[good])*response[good]**2/LBOX**3
        residual=abs(linear)**2-theory
        for state in q:
            coef=np.interp(kval,geo.centres,q[state],left=q[state][0],right=0.)
            power=np.where(good,-coef*residual,0.)
            for nmesh,(kt,kr) in both_kernels.items():
                correction[nmesh][state]+=(kt@power.ravel())[:,None]*kr[iz][None,:]*(1 if iz==0 else 2)
            if check_power is not None and state=='pre':
                check_power[:,:,iz][np.ix_(pick,pick)]=power
    out=dict(phase=np.array(phase),los=np.array(los),kmax=KMAX,
             coefficient=np.array('leave-one-phase-out high-resolution monopole; shared by both grids'),
             coefficient_k=geo.centres,growth=growth)
    if check_power is not None:
        xi=irfftn(check_power,s=(512,)*3,workers=workers)*512**3
        _,expected=bin_rppi(xi,512)
        error=np.max(abs(expected-correction[512]['pre']))
        np.testing.assert_allclose(expected,correction[512]['pre'],rtol=1e-8,atol=2e-13)
        out['direct_vs_fft_max_error']=error
        print('direct/FFT absolute difference',error,flush=True)
        del xi,check_power
    del delta;gc.collect()
    for state in q:
        out[f'{state}_coefficient']=q[state]
        suffix='' if los=='z' else '_LOSy'
        low_path=DATA/f'xi_lcv_{state}_Model1{suffix}_ph{phase}.npz'
        low=np.load(low_path)
        for nmesh in [512,1152]:
            raw=low['xirppi_raw'] if nmesh==512 else high[int(phase)][f'{state}_raw']
            out[f'n{nmesh}_{state}_raw']=raw
            out[f'n{nmesh}_{state}_lcv']=raw+correction[nmesh][state]
            out[f'n{nmesh}_{state}_correction']=correction[nmesh][state]
    out.update(elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    np.savez(dest,**out)
    print('saved',dest,'seconds',out['elapsed_seconds'],'peak GB',out['peak_rss_gb'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True)
    p.add_argument('--los',choices=['y','z'],required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(f'{int(a.phase):03d}',a.los,a.workers)
