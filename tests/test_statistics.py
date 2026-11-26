import numpy as np
import pytest
from nndr.statistics import (portmanteau_test,squared_residual_test,variance_matched_surrogate,
                             surrogate_test,block_bootstrap,bootstrap_confidence_interval)


def test_portmanteau_rotation():
    E=np.random.default_rng(4).normal(size=(2,500)); A=np.array([[2.,1.],[-1.,3.]])
    a=portmanteau_test(E,max_lag=5); b=portmanteau_test(A@E,max_lag=5)
    assert a['df']==20 and a['statistic']==pytest.approx(b['statistic'])
    assert 0<=a['pvalue']<=1


def test_portmanteau_detects_autocorrelation():
    E=np.random.default_rng(5).normal(size=(2,1000)); Z=E.copy()
    for i in range(1,len(Z.T)): Z[:,i]+=.8*Z[:,i-1]
    assert portmanteau_test(Z)['statistic']>portmanteau_test(E)['statistic']


def test_portmanteau_singular():
    with pytest.raises(ValueError): portmanteau_test(np.ones((2,30)))


def test_monte_carlo_ties():
    r=surrogate_test(2,[1,2,3]); assert r['pvalue']==.75 and r['percentile']==.5
    assert surrogate_test(4,[1,2,3])['pvalue']==.25
    assert surrogate_test(0,[1,2,3],alternative='less')['pvalue']==.25


def test_monte_carlo_nonfinite():
    with pytest.raises(ValueError): surrogate_test(1,[1,np.nan])


def series():
    F=np.array([[.7,.8],[0,.5]]); rng=np.random.default_rng(12); X=rng.normal(size=(2,500))
    for i in range(1,X.shape[1]): X[:,i]+=F@X[:,i-1]
    return F,X


def test_normal_null():
    F,X=series(); a=variance_matched_surrogate(F,X,seed=1); b=variance_matched_surrogate(F,X,seed=1)
    N=a['F']; S=a['stationary_covariance']; C=a['innovation_covariance']
    np.testing.assert_allclose(N@N.T,N.T@N,atol=1e-12)
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(N)),np.sort(np.linalg.eigvals(F)))
    np.testing.assert_allclose(S,N@S@N.T+C,atol=1e-12)
    assert np.trace(S)==pytest.approx(a['target_variance'])
    np.testing.assert_array_equal(a['X'],b['X'])


def test_normal_null_unscaled():
    F,X=series(); a=variance_matched_surrogate(F,X,match_variance=False)
    E=X[:,1:]-F@X[:,:-1]; E-=E.mean(axis=1,keepdims=True)
    np.testing.assert_allclose(a['innovation_covariance'],E@E.T/E.shape[1])
    assert a['variance_scale']==1


def test_normal_null_complex():
    F=np.array([[.5,.8],[-.2,.5]]); X=series()[1]
    a=variance_matched_surrogate(F,X); N=a['F']
    np.testing.assert_allclose(N@N.T,N.T@N,atol=1e-12)
    np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(N)),np.sort_complex(np.linalg.eigvals(F)))


def test_normal_null_unstable():
    with pytest.raises(ValueError): variance_matched_surrogate(np.eye(2),series()[1])


@pytest.mark.parametrize('method',['M2','M3'])
def test_squared_surrogate_refit(method):
    X=series()[1]; a=squared_residual_test(X,method=method,n_surrogates=4,max_lag=3,seed=42)
    b=squared_residual_test(X,method=method,n_surrogates=4,max_lag=3,seed=42)
    np.testing.assert_allclose(a['null'],b['null']); assert .2<=a['pvalue']<=1 and a['n_null']==4


def test_blocks():
    I=block_bootstrap(11,4,n_resamples=5,seed=4)
    assert I.shape==(5,11) and np.all((I>=0)&(I<11))
    assert np.all(np.diff(I[:,:4],axis=1)%11==1)
    np.testing.assert_array_equal(I,block_bootstrap(11,4,n_resamples=5,seed=4))


def test_bootstrap_ci():
    x=np.arange(101.); r=bootstrap_confidence_interval(50,x)
    assert r['lower']==pytest.approx(2.5) and r['upper']==pytest.approx(97.5) and r['bias']==0
    b=bootstrap_confidence_interval(50,x,method='bc'); assert b['lower']==pytest.approx(r['lower'])
    assert bootstrap_confidence_interval(80,x,method='bc')['lower']>r['lower']


def test_bootstrap_invalid():
    with pytest.raises(ValueError): block_bootstrap(3,4)
    with pytest.raises(ValueError): bootstrap_confidence_interval(0,[1,np.nan])


def test_unwrap_complex():
    from scipy.linalg import expm
    from nndr.statistics import unwrap_rates
    L=np.array([[-.3,4.],[-1.,-.3]])
    G=expm(2*L); r=unwrap_rates(G,L,dt=2)
    np.testing.assert_allclose(r['rates'],L,atol=1e-12)
    assert r['branch']!=0 and not r['ambiguous']


def test_unwrap_real_and_repeated():
    from scipy.linalg import expm
    from nndr.statistics import unwrap_rates
    L=np.array([[-.3,4.],[0,-.6]])
    np.testing.assert_allclose(unwrap_rates(expm(L),L)['rates'],L,atol=1e-12)
    with pytest.raises(ValueError): unwrap_rates(-np.eye(2),L)


def test_phase_preserves_cross_spectrum():
    from nndr.adequacy import phase_randomize
    X=series()[1]; Y=phase_randomize(X,seed=7)
    A=np.fft.rfft(X,axis=1); B=np.fft.rfft(Y,axis=1)
    np.testing.assert_allclose(A[:,None,:]*A[None,:,:].conj(),B[:,None,:]*B[None,:,:].conj(),atol=1e-9)
