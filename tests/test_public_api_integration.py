import numpy as np
import pytest
import nndr


def test_public_m3_high_level_and_legacy():
    F,Q=nndr.embedded_map(nndr.complex_generator(1,2,1.5),dimension=8,dt=.1,seed=4)
    direct=nndr.method3(F,dt=.1)
    res=nndr.analyze_operator(F,method='M3',dt=.1)
    assert np.allclose(res.Q,direct['Q'])
    assert np.allclose(res.r_hat,direct['r'])
    assert res.support==pytest.approx(nndr.reaction_support(direct['r']))
    assert res.R_A==pytest.approx(direct['diagnostics']['R_A'])
    assert res.extra['direction_identifiable']
    old=nndr.analyze_operator(F,method='M3',dt=.1,threshold='legacy')
    assert np.allclose(old.Q,nndr.method3_commutator(F)['Q'])
    assert old.R_A==pytest.approx(res.R_A)


def test_unidentifiable_direction_is_flagged():
    res=nndr.analyze_operator(.5*np.eye(4),method='M3')
    assert not res.extra['direction_identifiable']
    assert np.isnan(res.support)
