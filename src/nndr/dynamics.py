"""Dynamical response measures; explicitly distinguish maps and generators."""
from __future__ import annotations
import numpy as np
from .scientific import _map

__all__ = ['optimal_gain','variance_response','kreiss_constant','normal_surrogate','snapshot_gain']


def _propagator(A,t,continuous):
    if not np.isfinite(t) or t < 0:
        raise ValueError('time/horizon must be finite and nonnegative')
    if continuous:
        from scipy.linalg import expm
        return expm(A*float(t))
    if int(t)!=t:
        raise ValueError('discrete horizon must be an integer')
    return np.linalg.matrix_power(A,int(t))


def optimal_gain(A,t=1,*,continuous=False):
    """Maximum Euclidean norm gain at one specified horizon (not squared)."""
    return float(np.linalg.svd(_propagator(_map(A),t,continuous),compute_uv=False)[0])


def variance_response(A,t=1,*,continuous=False):
    """Trace of propagated unit isotropic covariance, ||P(t)||_F**2.

    Use continuous=True only for a continuous generator; otherwise A is a map.
    This is an impulse covariance response, not accumulated driven variance.
    """
    P = _propagator(_map(A),t,continuous)
    return float(np.sum(P**2))


def kreiss_constant(F,*,radii=None,n_angles=128):
    """Grid lower bound for the discrete Kreiss constant of a stable map.

    The supremum includes the limiting value 1 at infinity. The finite grid
    may underestimate it; this is not a certified maximization.
    """
    F=_map(F); rho=max(abs(np.linalg.eigvals(F)))
    if rho>1: return float('inf')
    if rho>=1: raise ValueError('unit-circle spectrum is not supported')
    if not isinstance(n_angles,(int,np.integer)) or n_angles<4:
        raise ValueError('n_angles must be an integer >= 4')
    radii=1+np.logspace(-4,2,48) if radii is None else np.asarray(radii,float)
    if radii.ndim!=1 or not radii.size or not np.all(np.isfinite(radii)) or np.any(radii<=1):
        raise ValueError('radii must be a nonempty vector strictly above one')
    I=np.eye(F.shape[0]); best=1.
    for r in radii:
        for a in np.linspace(0,2*np.pi,n_angles,endpoint=False):
            s=np.linalg.svd(r*np.exp(1j*a)*I-F,compute_uv=False)[-1]
            best=max(best,float((r-1)/s) if s else float('inf'))
    return best


def normal_surrogate(F):
    """Spectrum-matched real normal operator using real Schur coordinates.

    Does not match residual covariance or variance; those are simulation choices.
    Schur ordering is the library default, not an ordered Schur decomposition.
    """
    from scipy.linalg import schur
    F=_map(F); T,U=schur(F,output='real'); B=np.zeros_like(T); i=0
    while i<len(T):
        if i+1<len(T) and T[i+1,i]!=0:
            ev=np.linalg.eigvals(T[i:i+2,i:i+2]); a=float(ev.real.mean()); w=float(max(abs(ev.imag)))
            B[i:i+2,i:i+2]=[[a,w],[-w,a]]; i+=2
        else:
            B[i,i]=T[i,i]; i+=1
    return U@B@U.T


def snapshot_gain(A,t=1,*,continuous=False):
    """Leading singular pair and norm gain at one horizon (rebuttal R1.1).

    Overlap is absolute inner product of normalized initial and final vectors.
    Singular vectors need not be unique for tied leading singular values.
    """
    P=_propagator(_map(A),t,continuous); U,s,Vh=np.linalg.svd(P,full_matrices=False)
    u,v=U[:,0],Vh[0,:]
    return dict(gain=float(s[0]),u=u,v=v,overlap=float(abs(u@v)),
                unique=bool(len(s)==1 or s[0]-s[1]>1e-12*max(1.,s[0])))
