"""Estimator-matched quadrature for periodic Fourier and correlation bins.

Fourier weights integrate a tensor interpolation polynomial against the
ACTUAL discrete mode distribution, not a continuum k^2 dk dmu measure.
Configuration quadrature is Gaussian quadrature for each bin's discrete
Cartesian-lattice separation measure. Order=0 requests its exact support.
No cosmology, simulation field, or fitted alpha enters these rules.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import roots_legendre

LBOX=2000.0


def discrete_gauss(x, weight, order):
    x=np.asarray(x,dtype=float); weight=np.asarray(weight,dtype=float)
    unique,idx=np.unique(x,return_inverse=True)
    w=np.bincount(idx,weights=weight); w/=w.sum()
    if order==0 or len(unique)<=order: return unique,w
    # Stieltjes/Lanczos on a diagonal operator, with reorthogonalization.
    mid=(unique.min()+unique.max())/2; scale=(unique.max()-unique.min())/2
    z=(unique-mid)/scale
    vectors=[]; q=np.sqrt(w); alpha=[]; beta=[]
    for j in range(order):
        vectors.append(q)
        v=z*q; a=q@v; alpha.append(a)
        for u in vectors: v-=u*(u@v)
        for u in vectors: v-=u*(u@v)
        b=np.linalg.norm(v)
        if j<order-1:
            if b<1e-14: break
            beta.append(b); q=v/b
    jac=np.diag(alpha)+np.diag(beta,1)+np.diag(beta,-1)
    nodes,eig=np.linalg.eigh(jac)
    return mid+scale*nodes,eig[0]**2


@dataclass
class XiBins:
    nmesh: int
    order: int=6
    rmin: float=30.
    rmax: float=180.

    def __post_init__(self):
        centre=2+4*np.arange(50)
        self.rp_nominal,self.rt_nominal=np.meshgrid(centre,centre,indexing='ij')
        radius=np.hypot(self.rp_nominal,self.rt_nominal)
        self.mask=((radius>self.rmin)&(radius<self.rmax)).ravel()
        signed=np.arange(self.nmesh)
        distance=np.minimum(signed,self.nmesh-signed)*LBOX/self.nmesh
        xy=np.sqrt(distance[:,None]**2+distance[None,:]**2).ravel()
        self.rp=[]; self.rt=[]; self.weights=[]; self.index=[]
        radial=[]; transverse=[]
        for i in range(50):
            for dist,store in [(distance,radial),(xy,transverse)]:
                values=dist[(dist>=4*i)&(dist<4*(i+1))]
                if not len(values):raise ValueError('empty lattice bin')
                store.append(discrete_gauss(values,np.ones(len(values)),self.order))
        for ibin,original in enumerate(np.flatnonzero(self.mask)):
            ip,it=divmod(original,50)
            rp,wp=radial[ip]; rt,wt=transverse[it]
            pp,tt=np.meshgrid(rp,rt,indexing='ij')
            self.rp.extend(pp.ravel());self.rt.extend(tt.ravel())
            self.weights.extend((wp[:,None]*wt).ravel())
            self.index.extend([ibin]*pp.size)
        for name in ['rp','rt','weights','index']:
            setattr(self,name,np.asarray(getattr(self,name),dtype=int if name=='index' else float))
        self.nbins=int(self.mask.sum())
        np.testing.assert_allclose(self.average(np.ones(len(self.rp))),1,atol=2e-14)

    def average(self,values):
        return np.bincount(self.index,weights=self.weights*values,minlength=self.nbins)

    def broadband(self):
        r=np.hypot(self.rp,self.rt); mu=self.rp/r
        cols=[]
        for power in [-2,-1,0]:
            for ell in [0,2,4,6]:
                c=np.zeros(ell+1);c[ell]=1
                cols.append(self.average(r**power*np.polynomial.legendre.legval(mu,c)))
        return np.array(cols).T


def exact_fourier_modes(nmesh=512,dk=.005,nk=100,nmu=5):
    """Group equal (kx^2+ky^2,kz), retaining their exact multiplicities.

    The archived LCV estimator uses float32 |k| and float64 mu to assign
    bins. We follow it here, including mu=1 in the final wedge.
    """
    kf=2*np.pi/LBOX
    integers=np.fft.fftfreq(nmesh,d=1/nmesh).astype(int)
    transverse,number=np.unique((integers[:,None]**2+integers[None,:]**2).ravel(),return_counts=True)
    kval=[];muval=[];weight=[];index=[]
    edges=np.arange(nk+1)*dk;mu_edges=np.linspace(0,1,nmu+1)
    for iz in range(nmesh//2+1):
        k=np.sqrt((transverse.astype(float)+iz**2)*kf**2,dtype=np.float32)
        # A Python scalar / float32 array can compute in float32 under NumPy
        # scalar promotion. The estimator divides a float64 kz ARRAY by k.
        mu=np.divide(iz*kf,k.astype(float),out=np.zeros(len(k),dtype=float),where=k>0)
        ik=np.digitize(k,edges)-1
        im=np.clip(np.digitize(mu,mu_edges)-1,0,nmu-1)
        take=(ik>=0)&(ik<nk)
        kval.extend(k[take]);muval.extend(mu[take])
        weight.extend(number[take]*(1 if iz in [0,nmesh//2] else 2))
        index.extend(ik[take]*nmu+im[take])
    return dict(k=np.array(kval,dtype=float),mu=np.array(muval,dtype=float),
                weight=np.array(weight,dtype=float),index=np.array(index,dtype=int))


def lagrange(nodes,x):
    out=np.ones((len(x),len(nodes)))
    for j in range(len(nodes)):
        for k in range(len(nodes)):
            if k!=j:out[:,j]*=(x-nodes[k])/(nodes[j]-nodes[k])
    return out


class FourierBins:
    def __init__(self,order=6,nmesh=512,dk=.005,nk=100,nmu=5):
        self.order=order;self.nmesh=nmesh;self.dk=dk;self.nk=nk;self.nmu=nmu
        exact=exact_fourier_modes(nmesh,dk,nk,nmu)
        self.exact=exact;self.count=np.bincount(exact['index'],weights=exact['weight'],minlength=nk*nmu)
        self.mean_k=np.bincount(exact['index'],weights=exact['weight']*exact['k'],minlength=nk*nmu)/np.maximum(self.count,1)
        self.mean_mu=np.bincount(exact['index'],weights=exact['weight']*exact['mu'],minlength=nk*nmu)/np.maximum(self.count,1)
        gauss,_=roots_legendre(order)
        self.k=[];self.mu=[];self.weight=[]
        for b in range(nk*nmu):
            ik,im=divmod(b,nmu);take=exact['index']==b
            kn=(ik+.5+.5*gauss)*dk;mn=(im+.5+.5*gauss)/nmu
            kk,mm=np.meshgrid(kn,mn,indexing='ij')
            self.k.append(kk.ravel());self.mu.append(mm.ravel())
            if self.count[b]:
                lk=lagrange(kn,exact['k'][take]);lm=lagrange(mn,exact['mu'][take])
                weights=(lk.T@(exact['weight'][take,None]*lm))/self.count[b]
            else:weights=np.zeros((order,order))
            self.weight.append(weights.ravel())
        self.k=np.array(self.k);self.mu=np.array(self.mu);self.weight=np.array(self.weight)
        np.testing.assert_allclose(self.weight.sum(1)[self.count>0],1,atol=1e-12)

    def save(self,path):
        np.savez(path,order=self.order,nmesh=self.nmesh,dk=self.dk,nk=self.nk,nmu=self.nmu,
                 k=self.k,mu=self.mu,weight=self.weight,count=self.count,
                 mean_k=self.mean_k,mean_mu=self.mean_mu)

    @classmethod
    def load(cls,path):
        result=cls.__new__(cls)
        with np.load(path) as d:
            for key in d.files:setattr(result,key,np.array(d[key]))
        return result
