import numpy as np
import pytest
from nndr.estimation import adjacent_pairs, relative_ridge, shrink_operator, fit_relative_operator
from nndr.dynamics import optimal_gain, variance_response, kreiss_constant, normal_surrogate, snapshot_gain
from nndr.adequacy import prediction_skill, lag_structure_error, markov_rate_consistency, residual_autocorrelation, closure_error, phase_randomize, model_adequacy


def series():
    from nndr.synthetic_recovery import simulate_trajectory
    F=np.array([[.8,.5],[0.,.6]])
    return F,simulate_trajectory(F,samples=2000,seed=12)


def test_adjacent_pairs():
    X=np.arange(12).reshape(2,6); P,Y=adjacent_pairs(X)
    assert np.array_equal(P,X[:,:-1]) and np.array_equal(Y,X[:,1:])


def test_relative_scale():
    _,X=series(); F,info=fit_relative_operator(X,return_info=True)
    assert np.allclose(F,fit_relative_operator(1000*X),atol=1e-12)
    assert info['pairs']==1999
    assert info['condition']<=1+2/.003


def test_relative_penalty():
    X=np.array([[1.,2.],[3.,4.]])
    assert relative_ridge(X,.1)==pytest.approx(1.5)


def test_shrinkage():
    F=np.array([[.8,.5],[0.,.6]]); G=shrink_operator(F,.2)
    assert np.trace(G)==pytest.approx(np.trace(F))
    assert G[0,1]==pytest.approx(.4)


@pytest.mark.parametrize('lam',[-1,np.nan,np.inf])
def test_invalid_penalty(lam):
    with pytest.raises(ValueError): relative_ridge(np.eye(2),lam)


def test_gain():
    F=np.array([[.8,.5],[0.,.6]])
    assert optimal_gain(F,0)==pytest.approx(1)
    assert optimal_gain(F,3)==pytest.approx(np.linalg.norm(np.linalg.matrix_power(F,3),2))
    assert variance_response(F,0)==2
    assert variance_response(F,2)==pytest.approx(np.linalg.norm(F@F,'fro')**2)
    assert optimal_gain(-np.eye(2),.5,continuous=True)==pytest.approx(np.exp(-.5))
    with pytest.raises(ValueError): optimal_gain(F,.5)


def test_snapshot():
    F=np.array([[.8,.5],[0.,.6]]); d=snapshot_gain(F,3)
    assert np.allclose(np.linalg.matrix_power(F,3)@d['v'],d['gain']*d['u'])
    assert d['gain']==pytest.approx(optimal_gain(F,3))


def test_normal_surrogate():
    F=np.array([[.7,1.2,.3],[-.15,.7,.4],[0.,0.,.3]])
    G=normal_surrogate(F)
    assert np.allclose(G@G.T,G.T@G,atol=1e-12)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(F)),np.sort_complex(np.linalg.eigvals(G)))


def test_kreiss():
    assert kreiss_constant(.8*np.eye(2),radii=[1.1,2],n_angles=8)==1
    assert kreiss_constant(np.array([[.9,5],[0,.9]]),radii=[1.1,2],n_angles=8)>1


def test_prediction():
    F,X=series(); d=prediction_skill(X)
    assert d['train_pairs']+d['test_pairs']==X.shape[1]-1
    assert d['skill']>0


def test_exact_lag():
    F=np.diag([.8,.6]); X=np.column_stack([np.linalg.matrix_power(F,k)@np.ones(2) for k in range(10)])
    assert lag_structure_error(F,X,2)<1e-14
    m=markov_rate_consistency(X)
    assert all(v<1e-10 for v in m['relative_errors'].values())


def test_markov_singular():
    d=markov_rate_consistency(np.ones((2,10)))
    assert d['status'][1]=='singular_covariance'


def test_residuals():
    E=np.array([[1,-1,1,-1,1,-1],[1,1,1,1,1,1]],float)
    ac=residual_autocorrelation(E,max_lag=2)
    assert ac[0,0]==pytest.approx(-5/6)
    assert np.isnan(ac[1]).all()
    assert np.isnan(residual_autocorrelation(E,max_lag=2,squared=True)).all()


def test_closure():
    F=np.diag([.8,.7,.6]); Q=np.eye(3)[:,:2]
    assert closure_error(F,Q)==0
    F[2,0]=.2; assert closure_error(F,Q)>0


@pytest.mark.parametrize('T',[31,32])
def test_phase(T):
    X=np.random.default_rng(1).normal(size=(3,T)); Y=phase_randomize(X,seed=3)
    a,b=np.fft.rfft(X),np.fft.rfft(Y)
    assert np.allclose(a[:,None,:]*a[None,:,:].conj(),b[:,None,:]*b[None,:,:].conj())


def test_model():
    F,X=series(); d=model_adequacy(X,F,np.eye(2),max_lag=5)
    assert d['residual_acf'].shape==(2,5)
    assert d['closure_error']==0
