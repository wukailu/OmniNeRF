from utils.NMF.snmf import *
import numpy as np
from utils.NMF.base import *

if __name__ == '__main__':
    data = np.random.rand(50, 100)
    # data = np.array([[1.0, 0.0, 0.2],
    #                  [0.0, -1.0, 0.3]])

    mdl = SNMF(data, num_bases=27)

    mdl.factorize(niter=1000)

    # the reconstruction quality should be close to perfect
    rec = mdl.frobenius_norm()
    print(rec)
    # assert_almost_equal(0.0, rec, decimal=1)

    # and H is not allowed to have <0 values
    l = np.where(mdl.H < 0)[0]
    assert (len(l) == 0)

    W = mdl.W
    H = mdl.H
    approx = np.matmul(W, H)
    inf_norm = np.max(approx-data)
    print('matrix_shape', data.shape, 'W.shape', W.shape, 'H.shape', H.shape)
    print('data_max', np.max(data), 'diff_max', inf_norm)
    print('data_2norm', np.sum(np.abs(data**2)), 'diff_2norm', np.sum(np.abs(data-approx)**2))
    print('test passed')
