import numpy as np
import pytest
from nndr.thresholds import threshold_diagnostics as diag


def test_real_rate_formula():
    G = np.array([[.8, .7], [0, .3]])
    r = diag(G, .01)
    delta = abs(np.log(.8)-np.log(.3))/abs(np.log(.8)+np.log(.3))
    s = np.sqrt(1-delta**2)
    assert r['R_A'] == pytest.approx(1.4*np.sqrt((1-s)/s))
    assert r['Delta_rate'] == pytest.approx(delta)
    assert r['R_sigma'] == pytest.approx(.7/np.sqrt((1-.8**2)*(1-.3**2)))


def test_complex_and_distinct_criteria():
    # Exact exponential of L=[[-1,4],[-1,-1]] with dt=0.1.
    dt=.1
    G = np.exp(-dt)*(np.cos(2*dt)*np.eye(2)+np.sin(2*dt)*np.array([[0,2],[-.5,0]]))
    r=diag(G,dt)
    assert r['K'] == pytest.approx(.75)
    assert r['beta'] == pytest.approx(2)
    assert r['R_A'] == pytest.approx(.75*np.sqrt(np.sqrt(5)-1))
    assert r['R_A'] < 1 < r['R_sigma']
    assert r['R_A'] == pytest.approx(diag(G,1)['R_A'])


@pytest.mark.parametrize('G', [np.diag([.8,.3]), np.array([[.5,.4],[-.4,.5]]), .7*np.eye(2)])
def test_normal(G):
    r=diag(G)
    assert r['R_A'] == pytest.approx(0,abs=1e-14)
    assert r['R_sigma'] == pytest.approx(0,abs=1e-14)


def test_jordan_limit():
    G=np.array([[.7,.4],[0,.7]])
    r=diag(G)
    assert np.isinf(r['K'])
    assert r['R_A'] == pytest.approx(.4/(.7*2*np.sqrt(2)*abs(np.log(.7))))
    assert r['R_sigma'] == pytest.approx(.4/(1-.7**2))
    for eps in [1e-4,1e-7]:
        assert diag(G+np.array([[eps,0],[0,-eps]]))['R_A'] == pytest.approx(r['R_A'], rel=1e-5)
        assert diag(G+np.array([[0,0],[-eps**2/.4,0]]))['R_A'] == pytest.approx(r['R_A'], rel=1e-5)


@pytest.mark.parametrize('G',[np.diag([-.2,.5]), np.diag([-.2,-.5]), np.diag([0,.5])])
def test_inadmissible_log(G):
    r=diag(G)
    assert not r['rate_admissible'] and np.isnan(r['R_A'])
    assert r['R_sigma']==0


def test_unstable():
    r=diag(np.diag([1.,.5]))
    assert not r['stable_reduced']
    assert np.isnan(r['R_A']) and np.isnan(r['R_sigma'])


def test_random_identity_and_rotation():
    rng=np.random.default_rng(123)
    for _ in range(3000):
        G=rng.normal(size=(2,2))
        G *= rng.uniform(.05,.99)/max(abs(np.linalg.eigvals(G)))
        r=diag(G)
        assert (r['R_sigma']>1)==(r['sigmaGamma']>1)
        Q,_=np.linalg.qr(rng.normal(size=(2,2)))
        s=diag(Q.T@G@Q)
        assert s['R_sigma']==pytest.approx(r['R_sigma'],abs=1e-10,rel=1e-9)
        if r['rate_admissible']:
            assert s['R_A']==pytest.approx(r['R_A'],abs=1e-10,rel=1e-8)


def test_invalid_inputs():
    for G in [np.eye(3), np.eye(2,dtype=complex), np.full((2,2),np.nan)]:
        with pytest.raises(ValueError): diag(G)
    with pytest.raises(ValueError): diag(np.eye(2),0)
