"""Bin-averaged Vega with the output filter fixed in observed coordinates."""
import numpy as np
from fit_binned_xi import BinnedXiFitter
from target_bandpass import pass_power


class CommonTargetFitter(BinnedXiFitter):
    def __init__(self,nmesh,start=.35,small_scale_nl='dnl_arinyo',linear_template='vega'):
        super().__init__(nmesh,small_scale_nl=small_scale_nl,linear_template=linear_template)
        self.start=start
        # Initialize Vega once, then wrap the completed power before its
        # transform. Changing its fiducial P would also change the Arinyo
        # correction and would confound the test.
        BinnedXiFitter.predict(self,{})
        core=self.vega.models['lyaxlya'].Pk_core
        compute=core.compute
        cache={}
        def filtered(pk_lin,params,fast_metals=False):
            ap,at=self.vega.scale_params.get_ap_at(params,corr_name='LYAxLYA')
            key=bool(params['peak'])
            if key not in cache or cache[key][0]!=(ap,at):
                # Xi is evaluated at (ap*rp,at*rt). Applying W(ap*kpar,
                # at*kperp) before that dilation leaves W fixed at the
                # OBSERVED Fourier coordinate after it. Do not dilate a
                # measurement window along with the acoustic ruler.
                observed_k=np.sqrt((ap*core.k_par_grid)**2+(at*core.k_trans_grid)**2)
                cache[key]=((ap,at),pass_power(observed_k,self.start))
            return compute(pk_lin,params,fast_metals)*cache[key][1]
        core.compute=filtered
