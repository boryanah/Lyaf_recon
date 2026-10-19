"""One set of modes and isotropic held-out controls, measured in P and xi.

This uses only the existing Model 1 512-grid targets. It deliberately shares
the output filter and control coefficients with measure_common_target.py.
No field products are written, and the two sightlines remain paired by phase.
"""
import argparse
import gc
import importlib.util
from pathlib import Path
import resource
import time
import numpy as np
from target_bandpass import (OUT as COMMON, PREVIOUS, LBOX, IMAX, MODES,
                            compact_modes, pass_power, plane, add_plane,
                            load_kernels, save_atomic)
from highres_two_orientation_xirppi import linear_modes, post_path
from recon_paint_field import OUT_DIR, FOREST_RESPONSE
from ic_reference_spectrum import spectrum
from estimator_bins import FourierBins

OUT = Path('/pscratch/sd/b/boryanah/Lyaf_recon/matched_model_20260909')


def main(phase, los, workers=2):
    dest = OUT/f'measure_LOS{los}_ph{phase:03d}.npz'
    if dest.exists():
        print('exists', dest, flush=True); return
    cpu = time.process_time()
    prior = np.load(PREVIOUS/f'icmean_xi_LOS{los}_ph{phase:03d}.npz')
    spec = importlib.util.spec_from_file_location('lyaf_abacus_asdf',
        '/global/u1/b/boryanah/.local/perlmutter/python-3.11/lib/python3.11/site-packages/abacusnbody/data/asdf.py')
    ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
    import asdf
    ext.set_nthreads(workers); asdf.get_config().add_extension(ext.AbacusExtension())
    delta = linear_modes(f'{phase:03d}', los, 512, float(prior['growth']), workers)
    linear = np.ascontiguousarray(delta[np.ix_(MODES % 512, MODES % 512,
                                             np.arange(IMAX+1))])
    del delta; gc.collect()
    prepath = OUT_DIR/f'dF_ref_Model_1_LOS{los}_nmesh512_AbacusSummit_base_c000_ph{phase:03d}_z2.500.npy'
    fields = {'pre': compact_modes(prepath, 'node', workers),
              'post': compact_modes(post_path(f'{phase:03d}', los, 512, 512, OUT_DIR),
                                    'continuous', workers)}
    kernels = load_kernels(512); reference = spectrum(); bf, beta = FOREST_RESPONSE[1]
    result = dict(phase=phase, los=np.array(los), count=np.zeros(500),
                  coefficient_k=prior['coefficient_k'],
                  pre_coefficient=prior['pre_coefficient'],
                  post_coefficient=prior['post_coefficient'])
    for state in fields:
        for treatment in ['raw', 'lcv']:
            result[f'{state}_{treatment}_pk'] = np.zeros(500)
            result[f'{state}_{treatment}_xi'] = np.zeros((50, 50))
    for iz in range(IMAX+1):
        k, mu2 = plane(iz); good = (k > 0) & (k < .5)
        # Exactly the stored estimator's floating-point bin convention.
        kb = np.sqrt((MODES[:, None]**2+MODES[None, :]**2+iz**2)*(2*np.pi/LBOX)**2,
                     dtype=np.float32)
        mub = np.divide(iz*(2*np.pi/LBOX), kb.astype(float),
                    out=np.zeros_like(k), where=kb > 0)
        ik = np.searchsorted(np.arange(101)*.005, kb, side='right')-1
        im = np.minimum(np.searchsorted(np.linspace(0, 1, 6), mub, side='right')-1, 4)
        use = (kb < .5) & (im >= 0) & (ik >= 0)
        index = (ik*5+im)[use]; mult = 1 if iz == 0 else 2
        result['count'] += np.bincount(index, weights=np.full(index.size, mult), minlength=500)
        response = bf*(1+beta*mu2)
        mean = np.zeros_like(k)
        mean[good] = reference.plin(k[good])*response[good]**2/LBOX**3
        residual = abs(linear[:, :, iz]*response)**2-mean
        for state, modes in fields.items():
            q = np.interp(k, prior['coefficient_k'], prior[state+'_coefficient'],
                          left=prior[state+'_coefficient'][0], right=0.)
            for treatment in ['raw', 'lcv']:
                p = abs(modes[:, :, iz])**2
                if treatment == 'lcv':
                    p = p-q*residual
                p = np.where(good, p*pass_power(k, .35), 0.)
                tag = f'{state}_{treatment}'
                result[tag+'_pk'] += np.bincount(index, weights=mult*LBOX**3*p[use], minlength=500)
                add_plane(result[tag+'_xi'], p, kernels, iz)
    rule = FourierBins.load(PREVIOUS/'fourier_bins_order6.npz')
    np.testing.assert_array_equal(result['count'], rule.count)
    old = np.load(COMMON/f'common_target_LOS{los}_ph{phase:03d}.npz')
    for state in fields:
        for treatment in ['raw', 'lcv']:
            tag = f'{state}_{treatment}'
            result[tag+'_pk'] /= np.maximum(result['count'], 1)
            np.testing.assert_allclose(result[tag+'_xi'], old[f'w0.35_b512_n512_{tag}'],
                                       rtol=1e-10, atol=2e-13)
    result.update(cpu_seconds=time.process_time()-cpu,
                  peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    save_atomic(dest, **result)
    print('COMPLETE', dest, result['cpu_seconds'], result['peak_rss_gb'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', type=int, required=True)
    p.add_argument('--los', choices=['y', 'z'], required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args(); main(a.phase, a.los, a.workers)
