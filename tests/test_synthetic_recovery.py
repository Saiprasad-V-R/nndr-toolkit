import numpy as np
import pytest
from scipy.linalg import expm
from nndr.synthetic import real_generator, complex_generator
from nndr.synthetic_recovery import embedded_map, simulate_trajectory

@pytest.mark.parametrize('L', [real_generator(.5, 2., 3.), complex_generator(1., 2., 3.)])
@pytest.mark.parametrize('N', [2, 12])
def test_embedded_plane(L, N):
    F, Q = embedded_map(L, dimension=N, dt=.1, seed=3)
    np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(Q.T @ F @ Q, expm(.1*L), atol=1e-12)
    np.testing.assert_allclose(F @ Q, Q @ expm(.1*L), atol=1e-12)
    assert max(abs(np.linalg.eigvals(F))) < 1

def test_deterministic_simulation():
    F, Q = embedded_map(real_generator(.5, 2., 3.))
    X = simulate_trajectory(F, samples=100, seed=4)
    assert X.shape == (12, 100)
    np.testing.assert_array_equal(X, simulate_trajectory(F, samples=100, seed=4))
    assert not np.array_equal(X, simulate_trajectory(F, samples=100, seed=5))

@pytest.mark.parametrize('kwargs', [{'dimension':1},{'dt':0},{'background_rates':[1]},{'dt':2}])
def test_invalid_embedding(kwargs):
    with pytest.raises(ValueError):
        embedded_map(complex_generator(1., 2., 1.), **kwargs)

@pytest.mark.parametrize('kwargs', [{'samples':1},{'samples':20,'burn_in':-1},{'samples':20,'noise_std':0}])
def test_invalid_simulation(kwargs):
    with pytest.raises(ValueError):
        simulate_trajectory(.8*np.eye(2), **kwargs)

def test_unstable_map():
    with pytest.raises(ValueError):
        simulate_trajectory(1.1*np.eye(2), samples=20)
