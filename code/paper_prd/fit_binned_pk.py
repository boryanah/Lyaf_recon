"""BAO templates averaged over the exact periodic band-power estimator."""
from types import SimpleNamespace
import numpy as np
from estimator_bins import FourierBins
from referee_lcv_audit import PkmuModel,PkmuFitter,class_pk,source_path,bin_mask


class BinnedPkModel(PkmuModel):
    def __init__(self,cosmo,rule,mask,nbb=3,peak_only=False):
        mask=np.asarray(mask).ravel()
        self.rule=rule;self.mask=mask;self.peak_only=peak_only
        iw=np.tile(np.arange(5),int(rule.nk))[mask]
        super().__init__(cosmo,rule.mean_k[mask],rule.mean_mu[mask],iw,5,nbb=nbb)
        self.nodes=SimpleNamespace(cosmo=cosmo,k=rule.k[mask].ravel(),mu=rule.mu[mask].ravel())
        self.weights=rule.weight[mask]
        self.shape=self.weights.shape
        self.BB=np.column_stack([(iw==w)*np.sum(self.weights*rule.k[mask]**(-j),axis=1)
                                 for w in range(5) for j in range(nbb)])

    def peak(self,ap,at,beta,sp,st):
        nodes=self.nodes
        value=PkmuModel.peak(nodes,ap,at,beta,sp,st)
        if self.peak_only:
            kpar=nodes.k*nodes.mu/ap
            kk=np.hypot(kpar,nodes.k*np.sqrt(np.maximum(1-nodes.mu**2,0))/at)
            mm=kpar/kk
            value-=(1+beta*mm**2)**2*nodes.cosmo.plin(kk,wiggle=False)/(ap*at**2)
            value+=(1+beta*nodes.mu**2)**2*nodes.cosmo.plin(nodes.k,wiggle=False)
        return np.sum(self.weights*value.reshape(self.shape),axis=1)


def exact_template(rule,theta,cosmo,peak_only=False):
    ex=rule.exact;k=ex['k'].copy();k[k==0]=1e-8
    nodes=SimpleNamespace(cosmo=cosmo,k=k,mu=ex['mu'])
    value=PkmuModel.peak(nodes,*theta)
    if peak_only:
        ap,at,beta,sp,st=theta
        kpar=k*ex['mu']/ap;kk=np.hypot(kpar,k*np.sqrt(np.maximum(1-ex['mu']**2,0))/at);mm=kpar/kk
        value-=(1+beta*mm**2)**2*cosmo.plin(kk,wiggle=False)/(ap*at**2)
        value+=(1+beta*ex['mu']**2)**2*cosmo.plin(k,wiggle=False)
    value[ex['k']==0]=0
    return np.bincount(ex['index'],weights=ex['weight']*value,minlength=500)/np.maximum(rule.count,1)
