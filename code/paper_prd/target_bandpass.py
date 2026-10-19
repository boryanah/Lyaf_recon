"""Shared output-bandpass and exact-bin operations for target-grid tests."""
from pathlib import Path
import gc
import os
import numpy as np
from scipy.fft import rfftn

OUT = Path('/pscratch/sd/b/boryanah/Lyaf_recon/target_transfer_20260908')
PREVIOUS = OUT.parent / 'estimator_validation_20260908'
LBOX = 2000.
KF = 2 * np.pi / LBOX
KSTOP = .5
IMAX = int(np.ceil(KSTOP / KF))
MODES = np.arange(-IMAX, IMAX + 1)


def save_atomic(path,**values):
    """Publish complete compact products to fitting jobs on the shared FS."""
    path=Path(path)
    temporary=path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as stream:np.savez(stream,**values)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def pass_power(k, start=.35):
    """Squared amplitude of a C2 compact taper, unity below start, zero at .5."""
    u = np.clip((np.asarray(k) - start) / (KSTOP - start), 0, 1)
    amplitude = 1 - u**3 * (10 - 15*u + 6*u*u)
    return amplitude**2


def input_taper(k, nmesh):
    cut = (nmesh//2 - 1)*KF
    return .5 * (1 - np.tanh((np.abs(k)-.9*cut)/(.05*cut)))


def compact_modes(path, window, workers=4):
    """Crop normalized rFFT modes before assignment compensation; <15 GiB."""
    field = np.load(path, mmap_mode='r')
    n = field.shape[0]
    assert field.shape == (n,)*3 and n in [512, 1152]
    full = rfftn(field, workers=workers)
    full /= np.float32(n**3)
    del field
    result = np.ascontiguousarray(full[np.ix_(MODES % n, MODES % n,
                                             np.arange(IMAX+1))])
    del full
    gc.collect()
    if window == 'continuous':
        w = np.sinc(MODES/n)**3
        wz = np.sinc(np.arange(IMAX+1)/n)**3
    elif window == 'node':
        w = .75+.25*np.cos(2*np.pi*MODES/n)
        wz = .75+.25*np.cos(2*np.pi*np.arange(IMAX+1)/n)
    elif window == 'none':
        w = np.ones(len(MODES)); wz = np.ones(IMAX+1)
    else:
        raise ValueError(window)
    result /= (w[:,None,None]*w[None,:,None]*wz[None,None,:]).astype(np.float32)
    result[IMAX,IMAX,0] = 0
    return result


def load_kernels(nmesh):
    with np.load(PREVIOUS/f'xi_exact_bin_kernels_n{nmesh}.npz') as d:
        np.testing.assert_array_equal(d['modes'], MODES)
        return d['transverse'], d['radial']


def plane(iz):
    k2 = (MODES[:,None]*KF)**2+(MODES[None,:]*KF)**2+(iz*KF)**2
    k = np.sqrt(k2)
    mu2 = np.divide((iz*KF)**2,k2,out=np.zeros_like(k2),where=k2>0)
    return k, mu2


def add_plane(result, power, kernels, iz):
    transverse, radial = kernels
    result += (transverse @ power.ravel())[:,None]*radial[iz][None,:]*(1 if iz==0 else 2)


def translated_tsc_power(modes, nmesh, displacement):
    """Exact uniform-translation TSC / continuous-window power response.

    displacement is physical Mpc/h. A translation changes no acoustic scale.
    The local occupancy is exactly uniform for this regular shifted lattice.
    """
    fraction = (displacement/(LBOX/nmesh)) % 1
    omega = 2*np.pi*np.asarray(modes)/nmesh
    response = np.zeros_like(omega,dtype=complex)
    for j in [-1,0,1,2]:
        distance = abs(j-fraction)
        weight = (.75-distance**2 if distance<.5 else
                  .5*(1.5-distance)**2 if distance<1.5 else 0.)
        response += weight*np.exp(-1j*omega*j)
    return abs(response/np.sinc(np.asarray(modes)/nmesh)**3)**2
