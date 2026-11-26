"""Statistical controls for regularly sampled NNDR data.

These are building blocks, not a calibrated adequacy acceptance rule.
"""
from __future__ import annotations
import numpy as np
from .scientific import _series, _map, _plane
from .estimation import adjacent_pairs, fit_relative_operator
from .adequacy import phase_randomize, residual_autocorrelation
from .dynamics import normal_surrogate

__all__ = ['portmanteau_test', 'squared_residual_test', 'variance_matched_surrogate',
           'surrogate_test', 'block_bootstrap', 'bootstrap_confidence_interval']


def _count(n, name, minimum=1):
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(n)


def portmanteau_test(E, *, max_lag=10, fitted_df=0):
    """Multivariate adjusted portmanteau statistic on centered residuals.

    E is (coordinates, samples). Chi-square reference with df=d*d*h-fitted_df
    is asymptotic and assumes appropriate model/order and regularity conditions.
    Default fitted_df=0 is for externally supplied white-noise samples, not a
    calibrated p-value for estimated/projected NNDR residuals. Singular covariance
    is rejected rather than regularized silently.
    """
    from scipy.stats import chi2
    E = _series(E); d, T = E.shape
    h = _count(max_lag, 'max_lag'); k = _count(fitted_df, 'fitted_df', 0)
    df = d*d*h-k
    if h >= T or df <= 0: raise ValueError('invalid lag or degrees of freedom')
    Z = E-E.mean(axis=1, keepdims=True)
    C = Z@Z.T/T
    if np.linalg.matrix_rank(C) < d: raise ValueError('singular residual covariance')
    W = np.linalg.solve(np.linalg.cholesky(C), Z)
    stat = sum(T*T/(T-l)*np.linalg.norm(W[:,l:]@W[:,:-l].T/T,'fro')**2 for l in range(1,h+1))
    return dict(statistic=float(stat), df=df, pvalue=float(chi2.sf(stat, df)), samples=T,
                reference='asymptotic_chi_square_unadjusted_for_plane_selection')


def surrogate_test(observed, null, *, alternative='greater'):
    """Finite Monte Carlo upper/lower tail rank; ties counted conservatively.

    Valid p-value under exchangeability of observed and null statistics. No
    nonfinite results are silently discarded. Percentile is midrank among nulls.
    """
    null=np.asarray(null, float); observed=float(observed)
    if null.ndim!=1 or not null.size or not np.all(np.isfinite(null)) or not np.isfinite(observed):
        raise ValueError('observed and null statistics must be finite')
    if alternative=='greater': count=np.count_nonzero(null>=observed)
    elif alternative=='less': count=np.count_nonzero(null<=observed)
    else: raise ValueError('alternative must be greater or less')
    return dict(pvalue=float((1+count)/(1+len(null))), percentile=float((np.count_nonzero(null<observed)+.5*np.count_nonzero(null==observed))/len(null)),
                n_null=len(null), alternative=alternative)


def squared_residual_test(X, *, method='M3', max_lag=10, n_surrogates=199,
                          lam_rel=.003, shrinkage=0., seed=0):
    """Refit the same operator/plane on observed and phase-randomized series.

    Statistic: sum of squared sample autocorrelations of squared projected
    residuals over both plane coordinates and specified lags. Shared Fourier
    phases preserve the cross-periodogram, not marginal amplitude distributions.
    Inadmissible direction selection raises; no null realizations are dropped.
    This is a conditional surrogate comparison, not universal nonlinear detection.
    """
    from .scientific import method2, method3
    X=_series(X); B=_count(n_surrogates,'n_surrogates'); h=_count(max_lag,'max_lag')
    if h>=X.shape[1]-1: raise ValueError('max_lag too large')
    if method not in ('M2','M3'): raise ValueError('method must be M2 or M3')
    def statistic(Y):
        F=fit_relative_operator(Y,lam_rel=lam_rel,shrinkage=shrinkage)
        out=method3(F) if method=='M3' else method2(F,seed=seed)
        if not out.get('direction_identifiable',True): raise ValueError('directions not identifiable')
        P,T=adjacent_pairs(Y); E=out['Q'].T@(T-F@P)
        acf=residual_autocorrelation(E,max_lag=h,squared=True)
        if not np.all(np.isfinite(acf)): raise ValueError('undefined squared residual autocorrelation')
        return float(np.sum(acf**2))
    rng=np.random.default_rng(seed); observed=statistic(X)
    null=np.array([statistic(phase_randomize(X,seed=int(rng.integers(0,2**32)))) for _ in range(B)])
    return dict(observed=observed,null=null,**surrogate_test(observed,null))


def variance_matched_surrogate(F, X, *, match_variance=True, seed=0):
    """Stationary Gaussian spectrum-matched normal reference of length X.shape[1].

    Residual innovations are centered and estimated with divisor T-1 (number of
    pairs). Match total centered lag-0 variance in expectation, not each channel
    or each realization. Use a stationary Gaussian initial condition, no burn-in.
    The existing normal_surrogate uses library-default Schur order, not the
    modulus-ordered construction proposed in the rebuttal.
    """
    from scipy.linalg import solve_discrete_lyapunov
    F=_map(F); X=_series(X)
    if len(F)!=len(X): raise ValueError('dimension mismatch')
    if np.max(np.abs(np.linalg.eigvals(F)))>=1: raise ValueError('stable map required')
    P,Y=adjacent_pairs(X); E=Y-F@P; E-=E.mean(axis=1,keepdims=True)
    C=E@E.T/E.shape[1]; Fn=normal_surrogate(F)
    S=solve_discrete_lyapunov(Fn,C); S=(S+S.T)/2
    target=float(np.sum((X-X.mean(axis=1,keepdims=True))**2)/X.shape[1])
    base=float(np.trace(S))
    if base<=0: raise ValueError('zero stationary variance')
    scale=target/base if match_variance else 1.
    C*=scale; S*=scale
    def root(A):
        vals,U=np.linalg.eigh(A)
        if vals[0]<-1e-10*max(np.max(abs(vals)),np.finfo(float).tiny): raise ValueError('covariance is not positive semidefinite')
        return U*np.sqrt(np.maximum(vals,0))
    rng=np.random.default_rng(seed); out=np.empty_like(X)
    out[:,0]=root(S)@rng.standard_normal(len(F)); innovations=root(C)@rng.standard_normal((len(F),X.shape[1]-1))
    for j in range(1,X.shape[1]): out[:,j]=Fn@out[:,j-1]+innovations[:,j-1]
    return dict(X=out,F=Fn,innovation_covariance=C,stationary_covariance=S,variance_scale=scale,target_variance=target)


def block_bootstrap(n_samples, block_length, *, n_resamples=199, seed=0):
    """Circular moving-block bootstrap indices, shape (n_resamples,n_samples).

    Resample synchronized channels or window-level statistics. Do not fit a VAR
    across artificial block joins: resample aligned observed pairs instead.
    """
    n=_count(n_samples,'n_samples'); l=_count(block_length,'block_length'); B=_count(n_resamples,'n_resamples')
    if l>n: raise ValueError('block_length exceeds n_samples')
    rng=np.random.default_rng(seed); starts=rng.integers(n,size=(B,(n+l-1)//l))
    return ((starts[:,:,None]+np.arange(l))%n).reshape(B,-1)[:,:n]


def bootstrap_confidence_interval(observed, replicates, *, confidence=.95, method='percentile'):
    """Percentile or bias-corrected (BC, not BCa) interval from finite replicates.

    Reports empirical bias and bootstrap standard error. Coverage depends on
    resampling design; BC probabilities are clipped to finite Monte Carlo support.
    """
    from scipy.stats import norm
    x=np.asarray(replicates,float); observed=float(observed)
    if x.ndim!=1 or len(x)<2 or not np.all(np.isfinite(x)) or not np.isfinite(observed) or not 0<confidence<1:
        raise ValueError('finite scalar estimate, >=2 replicates and 0<confidence<1 required')
    a=(1-confidence)/2; q=np.array([a,1-a]); z0=0.
    if method=='bc':
        p=(np.count_nonzero(x<observed)+.5*np.count_nonzero(x==observed))/len(x)
        z0=float(norm.ppf(np.clip(p,.5/len(x),1-.5/len(x))))
        q=norm.cdf(2*z0+norm.ppf(q))
    elif method!='percentile': raise ValueError('method must be percentile or bc')
    lo,hi=np.quantile(x,q)
    return dict(lower=float(lo),upper=float(hi),bias=float(x.mean()-observed),se=float(x.std(ddof=1)),confidence=confidence,method=method,z0=z0,quantiles=q)


def unwrap_rates(G, reference, *, dt=1., tol=1e-10):
    """Closest real logarithm branch to a trusted 2D reference generator.

    For a distinct complex pair, choose the integer branch minimizing Frobenius
    distance to reference. Positive real distinct eigenvalues have the unique
    real log; repeated/negative/zero real eigenvalues are not resolved here.
    Not an ambient N-dimensional matching algorithm; cannot recover a missing
    aliased reference. An exact branch tie is flagged as ambiguous.
    """
    from scipy.linalg import logm
    G=_map(G,2); reference=_map(reference,2)
    if G.shape!=(2,2) or reference.shape!=(2,2) or not np.isfinite(dt) or dt<=0 or not np.isfinite(tol) or tol<=0:
        raise ValueError('two 2x2 matrices, positive dt and tol required')
    ev=np.linalg.eigvals(G).astype(complex)
    a=float(np.trace(G)/2); b=float(max(abs(ev.imag))); scale=max(np.linalg.norm(G,'fro'),np.finfo(float).tiny)
    if b<=tol*scale:
        if np.any(ev.real<=0) or abs(ev[0]-ev[1])<=tol*scale:
            raise ValueError('unresolved real/repeated spectrum')
        L=logm(G).real/dt
        return dict(rates=L,branch=0,ambiguous=False,error=float(np.linalg.norm(L-reference,'fro')))
    J=(G-a*np.eye(2))/b
    theta=np.arctan2(b,a); L0=(np.log(np.hypot(a,b))*np.eye(2)+theta*J)/dt
    D=2*np.pi*J/dt
    kstar=float(np.sum((reference-L0)*D)/np.sum(D*D)); k=int(np.rint(kstar))
    L=L0+k*D
    return dict(rates=L,branch=k,ambiguous=bool(abs(abs(kstar-k)-.5)<=tol),error=float(np.linalg.norm(L-reference,'fro')))


__all__.append('unwrap_rates')
