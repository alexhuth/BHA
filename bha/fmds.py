import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import scipy.linalg as la

from bha import BHA
from sampling import FarthestSampling
from mds import Scaling


class MF(BHA):
    def __init__(self, l, func, soft_p, sampling_method="farthest", interp_sq=True, **kwargs):
        super(MF, self).__init__(l=l, func=func, soft_p=soft_p,
                                   sampling_method=sampling_method, 
                                   interp_sq=interp_sq, **kwargs)
        # Make sure we get k-to-all sampling
        self._sampling_method = FarthestSampling(func, k_to_all=True)

    def fit(self, X):
        ## The following is adapted from BHA.fit
        n, d = X.shape
        self._n = n

        # If not computed, then compute the Nearest Neighbor graph
        # and the Laplace-Beltrami Operator.
        if self._M is None:
            self.preprocessing(X)
        # if "_Winds" not in dir(self) or self._Winds is None:
        Winds, F = self._sampling_method.sample(X, self._l)
        nonWinds = np.setdiff1d(np.arange(n), Winds)  # non-landmark points
        self._Winds = Winds
        self._nonWinds = nonWinds
        # else:
        #     raise NotImplementedError
        #     Winds = self._Winds
        #     nonWinds = self._nonWinds
        # #     compute kernel, W
        #     W = self.get_W(X, Winds)

        # store entire columns in F
        # F = self._func(X, self._Winds).T
        if self._interp_sq:
            self._F = F.T ** 2
        else:
            self._F = F.T
            
        # compute P 
        self.compute_P()

    def transform(self):
        MF = self._P.dot(self._F)
        return self.reconstruct(0.5*(MF + MF.T))

    def get_row(self, i):
        return self.reconstruct(0.5 * self._P[i,:].dot(self._F) + 0.5 * self._P.dot(self._F[:,i]))

    def get_rows(self, rows):
        return self.reconstruct(0.5 * self._P[rows,:].dot(self._F) + 0.5 * self._P.dot(self._F[:,rows]).T)

    def get_memory(self):
        if self._nnz_row == None:
            return (self._F.shape[0] * self._F.shape[1] + self._P.shape[0] * self._P.shape[1] ) * 8
        else:
            return (self._F.shape[0] * self._F.shape[1] + self._P.data.shape[0] + self._P.indices.shape[0]//2 + self._P.indptr.shape[0]//2) * 8

    def get_name(self):
        return 'mf'

    def reset(self, preprocessing=False):
        # reset the solution
        self._nonWinds = None
        self._Winds = None


class FMDS(Scaling):
    def __init__(self, k, mf=None, method='qr'):
        """
        :param k: the embedding dimension 
        :param mf: an MF method preprocessed
        :param method: 'qr'
        """
        super(FMDS, self).__init__(k)
        self._mf = mf
        self._method = method

    def preprocessing(self, pts, polys):
        self._mf.preprocessing(pts, polys)

    def fit(self, X):
        self._n = X.shape[0]
        # 1. Compute M and F matrices
        self._mf.fit(X)

        # 2. Apply the method and obtain Z
        if self._method == 'qr':
            self.Z = self._qr(X)
        elif self._method == 'ransvd':
            raise ValueError('Method not implemented: %s' % self._method)
        else:
            raise ValueError(self._method)

        return self.Z

    def _qr(self, X):
        M = self._mf._P
        F = self._mf._F
        n = X.shape[0]
        l = M.shape[1]

        # 1. Compute A=JS
        A = np.empty((n, 2*l))
        A[:, :l] = M - np.ones((n, 1)).dot(np.ones((1, n)).dot(M))/n
        A[:, l:] = F.T - np.ones((n, 1)).dot(np.ones((1, n)).dot(F.T))/n

        # 2. Compute A=QR
        Q, R = np.linalg.qr(A, mode='reduced')

        # 3. Compute Y=-1/4 RTR^T
        Y = np.empty((2*l, 2*l))
        Y[:l, :l] = R[:l, :l].dot(R[:l, l:].T) # top * full.T + full * top.T
        Y[:l, :l] += Y[:l, :l].T
        Y[:l, l:] = R[:l, :l].dot(R[l:, l:].T) # top * bottom.T
        Y[l:, :l] = Y[:l, l:].T                # bottom * top.T
        Y[l:, l:] = np.zeros(l)
        Y = -0.25*Y

        # 4. Truncated Eigendecomposition of Y=V D V^T with k eigenvalues
        D, V = la.eigh(Y, eigvals=(2*l - self._k, 2*l - 1))

        # 5. Compute Z = Q V D^1/2
        Z = Q.dot(V.dot(np.diag(np.sqrt(D))))

        return Z

    def get_memory(self):
        l = self._mf._P.shape[1]
        n = self._n
        return self._mf.get_memory() + (2*l*n + 4*l*l + self._k + self._k*2*l) * 8

    def get_name(self):
        return 'fmds'
