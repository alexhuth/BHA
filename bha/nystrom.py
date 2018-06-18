import numpy as np
import scipy.linalg as la

from sampling import RandomSampling, FarthestSampling
from base import SymMatrixApprox


class Nystrom(SymMatrixApprox):
    def __init__(self, k, func=None, rcond=None, sampling_method='random'):
        self._func = func
        self._k = k
        self._rcond = rcond
        if sampling_method=='random':
            self._sampling_method = RandomSampling(func, k_to_all=True)
        elif sampling_method=='farthest':
            self._sampling_method = FarthestSampling(func, k_to_all=True)
        else:
            raise ValueError('Unknown sampling method')

    def fit(self, X, y=None, chosen_inds=None):
        n = X.shape[0]

        if self._func is None:
            raise ValueError('function==None is not accepted.')

        if chosen_inds is None:
            chosen_inds, C = self._sampling_method.sample(X, self._k)
        else:
            C = self._func(X, chosen_inds, np.arange(n))

        W = C[chosen_inds, :]
        Wpinv = la.pinvh(W, rcond=self._rcond)
        self._n = n
        self._C = C
        self._idxs = chosen_inds
        self._Wpinv = Wpinv

    def get_row(self, i):
        return self._C[i, :].dot(self._Wpinv).dot(self._C.T)

    def get_rows(self, rows):
        return self._C[rows, :].dot(self._Wpinv).dot(self._C.T)

    def get_size(self):
        return self._n

    def get_memory(self):
        return (self._C.shape[0] * self._C.shape[1] + self._Wpinv.shape[0] * self._Wpinv.shape[1] ) * 8

    def get_name(self):
        return 'nystrom'


class MeshNystrom(Nystrom):
    def __init__(self, k, func=None, rcond=None, sampling_method='random'):
        super(MeshNystrom, self).__init__(k, func=func, rcond=rcond, sampling_method=sampling_method)
        if self._func is None:
            raise ValueError('function==None is not accepted.')

    def preprocessing(self, pts, polys):
        pass

    def fit(self, pts):
        n = pts.shape[0]

        chosen_inds, C = self._sampling_method.sample(pts, self._k)
        W = C[chosen_inds, :]
        Wpinv = la.pinvh(W, rcond=self._rcond)
        self._n = n
        self._C = C
        self._idxs = chosen_inds
        self._Wpinv = Wpinv
