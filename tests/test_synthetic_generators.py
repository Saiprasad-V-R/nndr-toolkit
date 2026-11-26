import numpy as np
import pytest
from nndr.synthetic import real_generator, complex_generator


@pytest.mark.parametrize('k', [0., 0.5, 2., 10.])
def test_real_generator(k):
    L = real_generator(0.5, 2., k)
    np.testing.assert_allclose(np.linalg.eigvals(L), [-0.5, -2.])
    assert abs(L[0, 1])/1.5 == pytest.approx(k)


@pytest.mark.parametrize('k', [0., 0.5, 2., 10.])
def test_complex_generator(k):
    L = complex_generator(1., 2., k)
    ev = np.linalg.eigvals(L)
    np.testing.assert_allclose(ev.real, -1.)
    np.testing.assert_allclose(np.abs(ev.imag), 2.)
    assert abs(L[0, 1]+L[1, 0])/4 == pytest.approx(k, abs=1e-14)


@pytest.mark.parametrize('fun,args', [(real_generator, (2, 1, 1)),
    (real_generator, (1, 1, 1)), (real_generator, (0, 2, 1)),
    (complex_generator, (1, 0, 1)), (complex_generator, (1, 2, -1)),
    (complex_generator, (1, 2, np.nan))])
def test_invalid_generator(fun, args):
    with pytest.raises(ValueError):
        fun(*args)
