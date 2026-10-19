"""The ensemble spectrum of the actual backscaled IC surrogate.

Do not substitute a fresh Boltzmann P(k,z_target) for this expectation.
Abacus generated the supplied density using its recorded z=1 spectrum and
ZD_Pk_sigma_ratio. The surrogate then multiplies that density by the growth
factor recorded in the LCV cache. Its expectation follows those operations.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from template_pk import TabulatedPk

OUT=Path('/pscratch/sd/b/boryanah/Lyaf_recon/estimator_validation_20260908')
CACHE=OUT/'ic_reference_spectrum.npz'


def build():
    import asdf
    spec=importlib.util.spec_from_file_location('lyaf_abacus_asdf',
        '/global/u1/b/boryanah/.local/perlmutter/python-3.11/lib/python3.11/site-packages/abacusnbody/data/asdf.py')
    ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
    asdf.get_config().add_extension(ext.AbacusExtension())
    path='/global/cfs/cdirs/desi/cosmosim/Abacus/ic/AbacusSummit_base_c000_ph000/ic_dens_N1152.asdf'
    with asdf.open(path,lazy_load=True) as f:
        table=f['CLASS_power_spectrum'];header=f['header']
        # Current numerical environment preserves the legacy ASDF table as
        # a tagged dictionary. Only its two small columns are materialized.
        k=np.array(table['columns'][0]['data'])
        power=np.array(table['columns'][1]['data'])
        ratio=float(header['ZD_Pk_sigma_ratio'])
        assert header['ZD_qPLT']==0 and header['ZD_Pk_smooth']==0
        growth=float(np.load(OUT/'matched_xi_LOSz_ph000.npz')['growth'])
        np.savez(CACHE,k=k,power_file=power,power_surrogate=power*(ratio*growth)**2,
                 sigma_ratio=ratio,growth_applied=growth,
                 growth_header=header['GrowthTable'][2.5],
                 file_redshift=header['ZD_Pk_file_redshift'],
                 source=np.array(path),description=np.array(__doc__))
    return CACHE


def spectrum():
    if not CACHE.exists():raise FileNotFoundError(f'Run ic_reference_spectrum.py first: {CACHE}')
    d=np.load(CACHE)
    return TabulatedPk(d['k'],d['power_surrogate'],d['power_surrogate'],np.nan,
                       label='exact IC-surrogate expectation (no BAO split)')


if __name__=='__main__':print(build())
