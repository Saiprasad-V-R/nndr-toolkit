import numpy as np
import pytest
from nndr import scientific as s
from nndr.synthetic import real_generator, complex_generator
from nndr.synthetic_recovery import embedded_map

@pytest.mark.parametrize('L',[real_generator(.5,2,3),complex_generator(1,2,1.5)])
def test_ratios_and_splitting(L):
    F,Q=embedded_map(L,dimension=2,dt=.1,seed=4)
    G=s.project_operator(F,Q)
    K=s.non_normality_index(G); d=s.eigenvalue_splitting(G); b=s.complex_splitting(G)
    kc=s.critical_K_real(d) if np.isfinite(d) else s.critical_K_complex(b)
    assert s.variance_response_ratio(G)==pytest.approx(K/kc)
    assert (s.one_step_ratio(G)>1)==(s.sigma_max(G)>1)
    assert s.eigenvector_conditioning(G)==pytest.approx(K+np.hypot(K,1))
    assert s.capture_fraction(F,Q)==pytest.approx(1)

@pytest.mark.parametrize('seed',range(5))
def test_order_exact(seed):
    rng=np.random.default_rng(seed); F=rng.normal(size=(2,2)); Q=np.linalg.qr(rng.normal(size=(2,2)))[0]
    P,c,ok=s.order_response_plane(F,Q)
    theta=np.linspace(0,np.pi,10001); v=np.array([np.cos(theta),np.sin(theta)]);u=np.array([-v[1],v[0]])
    assert c>=np.max(abs(np.sum(u*(F@v),axis=0)))-1e-12
    assert P[:,0]@F@P[:,1]==pytest.approx(c)
    assert np.allclose(P.T@P,np.eye(2)) and ok

@pytest.mark.parametrize('L',[real_generator(.5,2,3),complex_generator(1,2,1.5)])
def test_m3_order(L):
    F,Q=embedded_map(L,dimension=8,dt=.1,seed=4)
    out=s.method3(F)
    assert s.principal_angle(Q,out['Q'])<1e-5
    assert np.allclose(out['Q'][:,0],out['r']) and np.allclose(out['Q'][:,1],out['n'])
    assert out['direction_identifiable']
    assert s.variance_response_ratio(out['Gamma'])==pytest.approx(s.variance_response_ratio(Q.T@F@Q))

def test_m3_normal_flag():
    out=s.method3(np.eye(4)*.5)
    assert not out['plane_identifiable'] and not out['direction_identifiable']

def test_correlation_exact_and_rotation():
    rng=np.random.default_rng(8); X=rng.normal(size=(3,1000)); X[0]*=100; X[1]+=X[0]*.02
    Q=np.eye(3)[:,:2]; y=.001*X[0]+2*X[1]+.4*X[2]
    val,r=s.plane_correlation(X,Q,y,return_direction=True)
    assert val==pytest.approx(np.corrcoef(r@X,y)[0,1])
    assert val==pytest.approx(s.plane_correlation(X,Q@np.array([[0,1],[-1,0]]),y))
    assert np.linalg.norm(r)==pytest.approx(1)

def test_correlation_degenerate():
    X=np.arange(20.).reshape(1,-1); X=np.vstack([X,X,np.ones(20)]);Q=np.eye(3)[:,:2]
    assert s.plane_correlation(X,Q,X[0])==pytest.approx(1)
    assert np.isnan(s.plane_correlation(X,Q,np.ones(20)))
    assert np.isnan(s.plane_correlation(np.ones((3,20)),Q,np.arange(20)))

def test_scalar_diagnostics():
    F=np.diag([.5,.7,.8]);Q=np.eye(3)[:,:2]
    assert s.reaction_support(np.ones(4))==pytest.approx(1)
    assert s.reaction_support(np.array([1,0,0,0]))==pytest.approx(.5)
    assert np.isnan(s.reaction_support(np.zeros(4)))
    assert s.spectral_radius(F)==pytest.approx(.8)
    assert s.capture_fraction(F,Q)==pytest.approx(.7/.8)
    assert s.departure_from_normality(F)==pytest.approx(0,abs=1e-7)
    assert np.allclose(s.commutator(F),0)
    assert s.commutator_gap(F)==0
    assert s.relative_operator_error(F,F)==0
    assert s.numerical_abscissa(-F)==pytest.approx(-.5)
    assert s.condition_number(F)==pytest.approx(1.6)
    assert np.isinf(s.critical_K_real(0)) and np.isinf(s.critical_K_complex(0))

@pytest.mark.parametrize('x',[-1,1,np.nan])
def test_invalid_real_threshold(x):
    with pytest.raises(ValueError):s.critical_K_real(x)
