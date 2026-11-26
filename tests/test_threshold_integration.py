"""Run these tests after applying the update to the full toolkit."""
import numpy as np
import pytest
from nndr import analyze_operator, rolling_analyze, compute_indices_from_Q


def test_projection_and_legacy_switch():
    G=np.array([[.8,.7],[0,.3]])
    new=compute_indices_from_Q(G,np.eye(2),dt=.02)
    old=compute_indices_from_Q(G,np.eye(2),threshold='legacy')
    assert new['KKc']==new['R_A']
    assert old['R_A']==pytest.approx(new['R_A'])
    assert old['KKc']!=pytest.approx(new['KKc'])
    assert new['stable_full'] and new['stable_reduced']


def test_known_operator_api():
    G=np.array([[.5,.8],[-.2,.5]])
    res=analyze_operator(G,method='M2',dt=.02)
    assert res.ok
    assert res.R==res.R_A
    assert res.R_sigma==pytest.approx(compute_indices_from_Q(G,np.eye(2))['R_sigma'])
    assert res.diagnostics['spectrum_type']=='complex_pair'


def test_rolling_diagnostics_present():
    X=np.random.default_rng(4).normal(size=(3,100))
    rows=rolling_analyze(X,window=40,step=20,lam=.01,dt=.02)
    assert len(rows)==4
    assert all('R_A' in r and 'R_sigma' in r and 'rate_status' in r for r in rows)
