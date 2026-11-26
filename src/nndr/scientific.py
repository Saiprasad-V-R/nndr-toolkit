"""Public mathematical diagnostics for NNDR (real maps and observations)."""
from __future__ import annotations
import numpy as np
from .thresholds import threshold_diagnostics
from .methods import method1_eigenbasis_svd, method2_optimization, method3_commutator
from .reduce2d import gamma_diagnostics

__all__ = ['method1', 'method2', 'method3', 'order_response_plane',
           'reaction_support', 'mean_field', 'response_coordinate', 'input_coordinate',
           'project_operator', 'plane_correlation', 'eigenvalue_splitting',
           'complex_splitting', 'non_normality_index', 'eigenvector_conditioning',
           'critical_K_real', 'critical_K_complex', 'variance_response_ratio',
           'one_step_ratio', 'capture_fraction', 'spectral_radius', 'sigma_max',
           'departure_from_normality', 'commutator', 'commutator_gap',
           'principal_angle', 'relative_operator_error', 'numerical_abscissa',
           'condition_number']


def _real(x, name):
    x = np.asarray(x)
    if not np.isrealobj(x) or not np.all(np.isfinite(x)):
        raise ValueError(f'{name} must be finite and real')
    return x.astype(float)


def _map(F, minimum=1):
    F = _real(F, 'F')
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] < minimum:
        raise ValueError('F must be a nonempty square matrix of required dimension')
    return F


def _plane(Q, N):
    Q = _real(Q, 'Q')
    if Q.shape != (N, 2) or not np.allclose(Q.T @ Q, np.eye(2), atol=1e-10, rtol=1e-10):
        raise ValueError('Q must have shape (N,2) with orthonormal columns')
    return Q


def _series(X):
    X = _real(X, 'X')
    if X.ndim != 2 or min(X.shape) < 1:
        raise ValueError('X must have shape (channels,samples)')
    return X


def project_operator(F, Q):
    """Return Q.T @ F @ Q, for an orthonormal two-dimensional plane."""
    F = _map(F, 2); Q = _plane(Q, F.shape[0])
    return Q.T @ F @ Q


def reaction_support(r):
    """Eq. (61): L1(r)/(sqrt(N)*L2(r)); NaN for a zero direction."""
    r = _real(r, 'r')
    if r.ndim != 1 or not r.size: raise ValueError('r must be a nonempty vector')
    scale = np.max(np.abs(r))
    if scale == 0: return float('nan')
    r = r/scale
    return float(np.linalg.norm(r,1)/(np.sqrt(r.size)*np.linalg.norm(r)))


def mean_field(X):
    """Eq. (24): mean over channels, returning one value per sample."""
    return _series(X).mean(axis=0)


def response_coordinate(X, r):
    """Eq. (60): r.T @ X; r must be a unit direction."""
    X = _series(X); r = _real(r, 'r')
    if r.shape != (X.shape[0],) or not np.isclose(np.linalg.norm(r),1,rtol=1e-10,atol=1e-10):
        raise ValueError('r must be a unit vector matching the channel dimension')
    return r @ X


input_coordinate = response_coordinate


def plane_correlation(X, Q, observable=None, *, return_direction=False):
    """Eq. (25): exact maximum Pearson correlation over the plane.

    Default observable is the channel mean. Uses centered least squares,
    including rank-deficient plane coordinates. Undefined constant observable
    or rank-zero plane activity gives NaN. With return_direction=True returns
    (correlation, unit ambient direction), or a NaN direction if undefined.
    """
    X = _series(X); Q = _plane(Q, X.shape[0])
    y = mean_field(X) if observable is None else _real(observable, 'observable')
    if y.shape != (X.shape[1],) or y.size < 2: raise ValueError('observable must match samples, at least two')
    Z = (Q.T @ X).T; Z -= Z.mean(axis=0)
    y = y-y.mean(); yn = np.linalg.norm(y)
    a, _, rank, _ = np.linalg.lstsq(Z, y, rcond=None)
    if yn == 0 or rank == 0:
        score, direction = float('nan'), np.full(X.shape[0], np.nan)
    else:
        fit = Z @ a; score = float(np.clip(np.linalg.norm(fit)/yn, 0, 1))
        an = np.linalg.norm(a)
        direction = Q @ (a/an) if an > 0 else np.full(X.shape[0], np.nan)
    return (score, direction) if return_direction else score


def eigenvalue_splitting(Gamma, dt=1.):
    """Real-pair decay-rate splitting; NaN for other/inadmissible spectra."""
    return threshold_diagnostics(Gamma, dt)['Delta_rate']


def complex_splitting(Gamma, dt=1.):
    """Complex-pair beta=omega/gamma; NaN for other/inadmissible spectra."""
    return threshold_diagnostics(Gamma, dt)['beta']


def non_normality_index(Gamma):
    """Revised 2-D K; infinite for a nonnormal defective repeated pair."""
    return threshold_diagnostics(Gamma)['K']


def eigenvector_conditioning(Gamma):
    """Two-dimensional eigenvector condition number from K."""
    k = non_normality_index(Gamma)
    return float(np.hypot(1., k)+k)


def critical_K_real(delta):
    """Revised Eq. (B.4), real decay splitting 0 <= delta < 1."""
    delta=float(delta)
    if not np.isfinite(delta) or not 0 <= delta < 1: raise ValueError('require 0 <= delta < 1')
    if delta == 0: return float('inf')
    s=np.sqrt((1-delta)*(1+delta))
    return float(np.sqrt(s*(1+s))/delta)


def critical_K_complex(beta):
    """Revised Eq. (C.4), beta >= 0 (zero gives infinite limit)."""
    beta=float(beta)
    if not np.isfinite(beta) or beta < 0: raise ValueError('require finite beta >= 0')
    return float(np.sqrt(np.hypot(1.,beta)+1)/beta) if beta else float('inf')


def variance_response_ratio(Gamma, dt=1.):
    """R_A for a stable map with admissible principal logarithm."""
    return threshold_diagnostics(Gamma,dt)['R_A']


def one_step_ratio(Gamma):
    """R_sigma for a stable discrete map; NaN for unstable maps."""
    return threshold_diagnostics(Gamma)['R_sigma']


def spectral_radius(F):
    return float(np.max(np.abs(np.linalg.eigvals(_map(F)))))


def sigma_max(F):
    return float(np.linalg.svd(_map(F),compute_uv=False)[0])


def capture_fraction(F,Q):
    """Reduced-to-full largest singular value ratio; NaN for zero operator."""
    G=project_operator(F,Q); s=sigma_max(F)
    return float(sigma_max(G)/s) if s else float('nan')


def departure_from_normality(F):
    """Henrici departure; use stable 2-D formula where available.

    General-dimensional norm subtraction can lose relative accuracy near normality.
    """
    F=_map(F)
    if F.shape == (2,2): return threshold_diagnostics(F)['dF']
    scale=np.max(np.abs(F))
    if scale == 0: return 0.
    A=F/scale
    return float(scale*np.sqrt(max(0.,np.sum(A*A)-np.sum(np.abs(np.linalg.eigvals(A))**2))))


def commutator(F):
    """Eq. (37): F F.T - F.T F."""
    F=_map(F);return F@F.T-F.T@F


def commutator_gap(F):
    """Minimum separation of the two extreme eigenvalues from the complement.

    For N=2 the complementary subspace is empty, so the gap is infinite.
    This is a plane gap, not the gap between the two extreme eigenvalues.
    """
    ev=np.linalg.eigvalsh(commutator(_map(F,2)))
    return float(min(ev[1]-ev[0],ev[-1]-ev[-2])) if ev.size>2 else float('inf')


def principal_angle(Q,P):
    """Eq. (23): largest principal angle in degrees, not radians."""
    Q=np.asarray(Q)
    if Q.ndim!=2: raise ValueError('Q must be a matrix')
    Q=_plane(Q,Q.shape[0]);P=_plane(P,Q.shape[0])
    return float(np.degrees(np.arccos(np.clip(np.linalg.svd(Q.T@P,compute_uv=False)[-1],0,1))))


def relative_operator_error(Fhat,F):
    """Eq. (22): relative Frobenius operator error."""
    F=_map(F);Fhat=_map(Fhat)
    if Fhat.shape != F.shape: raise ValueError('operators must have identical shapes')
    norm=np.linalg.norm(F,'fro')
    return float(np.linalg.norm(Fhat-F,'fro')/norm) if norm else float('nan')


def numerical_abscissa(L):
    """Maximum eigenvalue of (L+L.T)/2; L is a continuous generator."""
    L=_map(L);return float(np.linalg.eigvalsh((L+L.T)/2)[-1])


def condition_number(F):
    """Euclidean condition number of the operator (not eigenbasis)."""
    return float(np.linalg.cond(_map(F)))


def order_response_plane(F,Q,*,tol=1e-12):
    """Solve manuscript Eq. (40) exactly inside an orthonormal plane.

    Return (ordered Q=[r,n], coupling, direction_identifiable). The latter
    excludes rotationally degenerate cases; the simultaneous sign is arbitrary.
    """
    G=project_operator(F,Q); Q=np.asarray(Q,dtype=float)
    a,b=G[0];c,d=G[1]
    k=(c-b)/2; x=(c+b)/2; y=(d-a)/2
    amp=np.hypot(x,y)
    # v=(cos theta,sin theta), u=(-sin theta,cos theta).
    # u.T G v = k+x*cos(2 theta)+y*sin(2 theta).
    sign=1. if k>=0 else -1.
    theta=.5*np.arctan2(sign*y,sign*x) if amp else 0.
    v=np.array([np.cos(theta),np.sin(theta)])
    u=sign*np.array([-v[1],v[0]])
    P=Q@np.column_stack([u,v])
    scale=max(1.,np.linalg.norm(G,2))
    return P,float(abs(k)+amp),bool(amp>tol*scale and abs(k)>tol*scale)


def _with_diagnostics(out, dt):
    out["legacy_okG"] = out["okG"]
    out["diagnostics"] = threshold_diagnostics(out["Gamma"],dt)
    out["okG"] = out["diagnostics"]["rate_admissible"]
    return out


def method1(F,*,dt=1.,**kwargs):
    """M1 historical eigenbasis-SVD baseline."""
    return _with_diagnostics(method1_eigenbasis_svd(_map(F,2),**kwargs),dt)


def method2(F,*,dt=1.,**kwargs):
    """M2 alternating directed-coupling optimization (not global certificate)."""
    return _with_diagnostics(method2_optimization(_map(F,2),**kwargs),dt)


def method3(F,*,tol=1e-12,dt=1.):
    """M3 commutator plane with exact internal ordering, Eqs. (37)-(42).

    For an unidentifiable plane the basis is still returned but flagged.
    Historical raw commutator basis remains in method3_commutator().
    """
    F=_map(F,2); out=method3_commutator(F,tol=tol)
    Q0=out['Q']; Q, coupling, direction_ok=order_response_plane(F,Q0,tol=tol)
    gap=commutator_gap(F); plane_ok=bool(out['okB'] and gap>tol)
    G=Q.T@F@Q; tr,ev,ok=gamma_diagnostics(G)
    out.update(Q_comm=Q0,Q=Q,r=Q[:,0],n=Q[:,1],Gamma=G,trG=tr,evG=ev,okG=ok,
               coupling=coupling,commutator_gap=gap,plane_identifiable=plane_ok,
               direction_identifiable=bool(plane_ok and direction_ok))
    return _with_diagnostics(out,dt)
