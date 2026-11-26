import numpy as np
import pytest
import nndr
from nndr.ridge import ridge_fit_F,stack_VAR_pairs_from_trajectories


def test_fit_relative_high_level():
    F=np.array([[.8,.5],[0.,.6]])
    X=nndr.simulate_trajectory(F,samples=500,seed=17)
    expected=nndr.fit_relative_operator(X,lam_rel=.003,shrinkage=.04)
    assert np.allclose(nndr.fit_operator(X,lam_rel=.003,shrinkage=.04),expected)
    a=nndr.analyze(X,method='M3',lam_rel=.003,shrinkage=.04)
    assert np.allclose(a.Fhat,expected)
    assert np.allclose(nndr.analyze(1000*X,method='M3',lam_rel=.003,shrinkage=.04).Fhat,expected)
    assert len(nndr.rolling_analyze(X,window=200,step=100,lam_rel=.003))==4
    with pytest.raises(ValueError):nndr.fit_operator(X,lam=.01,lam_rel=.003)


def test_adjacent_and_legacy_fit():
    X=np.array([[3.,1.,2.,4.],[1.,2.,3.,1.]])
    P,Y=nndr.adjacent_pairs(X)
    assert np.allclose(nndr.fit_operator(X,lam=.1),ridge_fit_F(P,Y,.1))
    P,Y=stack_VAR_pairs_from_trajectories([X],x0=np.zeros(2))
    assert np.allclose(nndr.fit_operator(X,lam=.1,legacy_pairs=True),ridge_fit_F(P,Y,.1))
    assert np.allclose(nndr.analyze(X,lam=.1,threshold='legacy').Fhat,ridge_fit_F(P,Y,.1))
