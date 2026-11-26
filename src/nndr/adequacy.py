"""Local predictor adequacy measures (rebuttal R1.3).

These are descriptive diagnostics, not a calibrated pass/fail screen.
"""
from __future__ import annotations
import numpy as np
from .scientific import _series,_map,_plane
from .estimation import adjacent_pairs,relative_ridge,fit_relative_operator

__all__=['prediction_skill','lag_structure_error','markov_rate_consistency','residual_autocorrelation','closure_error','phase_randomize','model_adequacy']


def prediction_skill(X,*,frac_train=.8,lam_rel=.003,shrinkage=0.):
    """Chronological held-out skill 1-MSE(full)/MSE(diagonal VAR1).

    Both models are fitted on identical training pairs and use the same ridge
    penalty. The diagonal baseline uses the same trace-preserving shrinkage
    convention. No preprocessing is fitted on the held-out segment.
    """
    P,Y=adjacent_pairs(X); n=int(frac_train*P.shape[1])
    if not 0<frac_train<1 or n<2 or n>=P.shape[1]: raise ValueError('insufficient train/test pairs')
    train=np.asarray(X)[:,:n+1]; F,info=fit_relative_operator(train,lam_rel=lam_rel,shrinkage=shrinkage,return_info=True)
    den=np.sum(P[:,:n]**2,axis=1)+info['lam']
    d=np.divide((P[:,:n]*Y[:,:n]).sum(axis=1),den,out=np.zeros(len(P)),where=den>0)
    d=(1-shrinkage)*d+shrinkage*d.mean()
    E=Y[:,n:]-F@P[:,n:]; Ed=Y[:,n:]-d[:,None]*P[:,n:]
    mse=float(np.mean(E**2)); base=float(np.mean(Ed**2))
    return dict(skill=1-mse/base if base>0 else np.nan,mse=mse,baseline_mse=base,train_pairs=n,test_pairs=P.shape[1]-n)


def _lag_pairs(X,lag):
    X=_series(X)
    if not isinstance(lag,(int,np.integer)) or not 0<lag<X.shape[1]: raise ValueError('invalid integer lag')
    return X[:,:-lag],X[:,lag:]


def lag_structure_error(F,X,lag):
    """Relative covariance error ||C_lag-F**lag C0||_F/||C_lag||_F.

    Both uncentered covariances use exactly the same lagged pairs; center X
    before calling if centered covariance is required.
    """
    F=_map(F); P,Y=_lag_pairs(X,lag)
    C0=P@P.T; Ct=Y@P.T; den=np.linalg.norm(Ct,'fro')
    return float(np.linalg.norm(Ct-np.linalg.matrix_power(F,lag)@C0,'fro')/den) if den else np.nan


def markov_rate_consistency(X,*,lags=(1,2,4),dt=1.,imag_tol=1e-8):
    """Compare principal-log lag maps, reporting inadmissible lags as NaN.

    Unregularized lag fits as in R1.3(c). No branch unwrapping is attempted;
    use only where aliasing is excluded. Singular covariance is flagged.
    """
    from scipy.linalg import logm
    if not np.isfinite(dt) or dt<=0: raise ValueError('dt must be positive')
    lags=tuple(lags)
    if not lags or lags[0]!=1 or len(set(lags))!=len(lags): raise ValueError('lags must be unique, starting with 1')
    rates={}; status={}
    for lag in lags:
        P,Y=_lag_pairs(X,lag); C=P@P.T
        if np.linalg.matrix_rank(C)<len(C): rates[lag]=None;status[lag]='singular_covariance';continue
        F=np.linalg.solve(C,(Y@P.T).T).T; ev=np.linalg.eigvals(F)
        if np.any((np.abs(ev.imag)<=imag_tol)&(ev.real<=0)):
            rates[lag]=None;status[lag]='no_real_principal_log';continue
        L=logm(F)/(lag*dt)
        if np.linalg.norm(L.imag)>imag_tol*max(1.,np.linalg.norm(L.real)):
            rates[lag]=None;status[lag]='complex_log';continue
        rates[lag]=L.real;status[lag]='ok'
    ref=rates[1]; den=np.linalg.norm(ref,'fro') if ref is not None else 0.
    errors={lag:float(np.linalg.norm(L-ref,'fro')/den) if L is not None and den else np.nan for lag,L in rates.items()}
    return dict(relative_errors=errors,status=status,rates=rates)


def residual_autocorrelation(E,*,max_lag=10,squared=False):
    """Per-coordinate centered autocorrelations at lags 1..max_lag.

    Constant coordinates return NaN. Squared residuals screen nonlinear
    dependence; this measure alone is not a statistical significance test.
    """
    E=_series(E)
    if not isinstance(max_lag,(int,np.integer)) or not 0<max_lag<E.shape[1]: raise ValueError('invalid max_lag')
    Z=E**2 if squared else E.copy(); Z-=Z.mean(axis=1,keepdims=True); den=np.sum(Z**2,axis=1)
    return np.column_stack([np.divide(np.sum(Z[:,:-l]*Z[:,l:],axis=1),den,out=np.full(len(Z),np.nan),where=den>0) for l in range(1,max_lag+1)])


def closure_error(F,Q):
    """Relative invariant-plane leakage ||FQ-Q(Q.T FQ)||_F/||FQ||_F."""
    F=_map(F,2);Q=_plane(Q,len(F)); Z=F@Q; den=np.linalg.norm(Z,'fro')
    return float(np.linalg.norm(Z-Q@(Q.T@Z),'fro')/den) if den else np.nan


def phase_randomize(X,*,seed=0):
    """Multivariate Fourier surrogate with a shared phase at each frequency.

    Preserves cross-periodograms and channel means; not amplitude distributions.
    """
    X=_series(X); Z=np.fft.rfft(X,axis=1); rng=np.random.default_rng(seed)
    phase=rng.uniform(0,2*np.pi,Z.shape[1]);phase[0]=0
    if X.shape[1]%2==0: phase[-1]=0
    return np.fft.irfft(Z*np.exp(1j*phase)[None,:],n=X.shape[1],axis=1)


def model_adequacy(X,F,Q,*,dt=1.,lags=(2,4),max_lag=10,lam_rel=.003,shrinkage=0.,frac_train=.8):
    """Return descriptive adequacy measures; no automatic accept/reject decision.

    Residual dependence uses supplied full-window F; held-out skill refits F
    on training data only. User must calibrate screening thresholds separately.
    """
    F=_map(F,2); Q=_plane(Q,len(F)); P,Y=adjacent_pairs(X); E=Q.T@(Y-F@P)
    return dict(prediction=prediction_skill(X,frac_train=frac_train,lam_rel=lam_rel,shrinkage=shrinkage),
                lag_errors={l:lag_structure_error(F,X,l) for l in lags},
                markov=markov_rate_consistency(X,dt=dt),
                residual_acf=residual_autocorrelation(E,max_lag=max_lag),
                squared_residual_acf=residual_autocorrelation(E,max_lag=max_lag,squared=True),
                closure_error=closure_error(F,Q))
