"""Vega correlation model integrated over each actual lattice bin.

The model and all 12 additive broadband columns undergo the same averaging.
Nominal bin centers determine the data cut and borrowed covariance indices;
they are not substituted for the separations when evaluating the model.
"""
import time
from pathlib import Path
import sys
import numpy as np
from vega.coordinates import RtRpCoordinates
from vega.minimizer import Minimizer
from estimator_bins import XiBins
from fit_lcv_xirppi_vega import setup_vega,covariance_and_mask

PARS=('ap','at','bias_LYA','beta_LYA','sigmaNL_par','sigmaNL_per')


class BinnedXiFitter:
    def __init__(self,nmesh,order=6,small_scale_nl='dnl_arinyo',rmin=30.,rmax=180.,linear_template='vega'):
        self.rule=XiBins(nmesh,order,rmin,rmax)
        self.vega,rp,rt,radius=setup_vega(small_scale_nl=small_scale_nl)
        if linear_template=='class' or linear_template.startswith('ic_'):
            sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
            if linear_template=='class':
                from template_pk import class_pk
                theory=class_pk(z=2.5)
            else:
                from ic_bao_template import load_template
                theory=load_template(linear_template.removeprefix('ic_'))
            self.vega.fiducial['pk_full']=theory.plin(self.vega.fiducial['k'])
            self.vega.fiducial['pk_smooth']=theory.plin(self.vega.fiducial['k'],wiggle=False)
            self.vega.fiducial['z_fiducial']=2.5
        elif linear_template!='vega':raise ValueError(linear_template)
        self.mask,self.icov=covariance_and_mask(rp,rt,radius,rmin,rmax)
        np.testing.assert_array_equal(self.mask,self.rule.mask)
        coords=RtRpCoordinates(rp_min=0.,rp_max=200.,rt_max=200.,rp_nbins=50,rt_nbins=50,
                                rp_grid=self.rule.rp,rt_grid=self.rule.rt,z_eff=2.5)
        # Remove Vega's zero-valued built-in broadband at quadrature nodes;
        # we profile the exact same basis after applying the bin average.
        self.vega.corr_items['lyaxlya'].config.remove_section('broadband')
        self.vega.corr_items['lyaxlya'].init_coordinates(coords)
        self.initialized=False
        b=self.rule.broadband()
        norm=np.sqrt(np.diag(b.T@self.icov@b))
        self.basis=b/norm
        self.icov_b=self.icov@self.basis
        self.normal=self.basis.T@self.icov_b
        self.sample={group:{k:v for k,v in values.items() if k in PARS}
                     for group,values in self.vega.sample_params.items()}

    def predict(self,params):
        full=self.vega.params.copy();full.update(params)
        values=self.vega.compute_model(full,run_init=not self.initialized)['lyaxlya']
        self.initialized=True
        if len(values)!=len(self.rule.rp):raise ValueError('Vega returned wrong coordinate layout')
        return self.rule.average(values)

    def fit(self,data):
        data=np.asarray(data)
        if data.shape==(50,50):data=data.T.ravel()[self.mask]
        if data.shape!=(self.rule.nbins,):raise ValueError(data.shape)
        def objective(params):
            residual=self.predict(params)-data
            rhs=self.icov_b.T@residual
            nuisance=np.linalg.solve(self.normal,rhs)
            # Subtract the profiled component explicitly to avoid cancellation.
            residual-=self.basis@nuisance
            return float(residual@self.icov@residual)
        start=time.monotonic()
        minimizer=Minimizer(objective,self.sample)
        minimizer.minimize()
        # Default Minuit tolerance is too loose for sub-0.01% noiseless
        # closure tests under this borrowed covariance normalization.
        minimizer.minuit.tol=1e-6
        minimizer.minuit.migrad(ncall=100000)
        if not minimizer.fmin.is_valid:raise RuntimeError(str(minimizer.fmin))
        result={k:float(minimizer.values[k]) for k in PARS}
        result.update(chi2=objective(minimizer.values),edm=float(minimizer.fmin.edm),
                      valid=bool(minimizer.fmin.is_valid),at_limit=bool(minimizer.fmin.has_parameters_at_limit),
                      elapsed_seconds=time.monotonic()-start)
        return result
