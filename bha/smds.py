import numpy as np
import scipy.sparse.linalg as sla
import scipy.sparse as sparse
import scipy.linalg as la

from base import SymMatrixApprox
from sampling import RandomSampling, FarthestSampling
from mds import Scaling
from cortex.polyutils import Surface


class SA(SymMatrixApprox):
    def __init__(self, l, m_eig, func=None, sampling_method='farthest', interp_sq=False):
        self._l = l
        self._me = m_eig
        self._func = func
        self._set_sampling_method(sampling_method, func)
        self._interp_sq = interp_sq  # interpolate squared distance, then sqrt?

    def preprocessing(self, pts, polys):
        # Compute Laplacian
        self._pts = pts
        self._polys = polys
        self._surface = Surface(pts, polys)

        B, D, lapW, lapV = self._surface.laplace_operator
        npt = len(D)
        Dinv = sparse.dia_matrix((D**-1, [0]), (npt, npt))  # construct Dinv
        self._L = Dinv.dot(lapV - lapW)
        self._L.tocsc().sort_indices()

        # extract the eigenvalues of the Laplacian
        self.Lambda, self.Phi = sla.eigs(self._L, k=self._me, which='SM')
        self.Lambda = np.real(self.Lambda)
        self.Phi = np.real(self.Phi)

    def fit(self, X):
        self._n = X.shape[0]
        # Sample the points and compute the matrix D
        inds, D = self._sampling_method.sample(X, self._l)
        noninds = np.setdiff1d(np.arange(self._n), inds)  # non-landmark points
        self._inds = inds
        self._noninds = noninds
        if self._interp_sq:
            D **= 2

        D = (D +D.T) / 2.0

        # Compute Interpolator matrix M
        A = sparse.csr_matrix((np.ones((self._l,)), (inds, np.arange(self._l))), shape=(self._n, self._l))
        APhi = A.T.dot(self.Phi)
        B = APhi.T.dot(APhi)
        M = np.linalg.solve(np.diag(self.Lambda) + B, APhi.T)
        self._alpha = M.dot(D).dot(M.T)

    def _set_sampling_method(self, sampling_method, func):
        if sampling_method=='random':
            self._sampling_method = RandomSampling(func, k_to_all=False)
        elif sampling_method=='farthest':
            self._sampling_method = FarthestSampling(func, k_to_all=False)
        else:
            raise ValueError('Unknown sampling method')

    def _reconstruct(self, approx):
        """Reconstruct the data from the approximation. Takes the square root
        of the approximation if we are approximating the squared matrix.
        """
        if self._interp_sq:
            return np.sqrt(np.clip(approx, 0, np.inf))
        else:
            return approx

    def transform(self):
        Kmanifold = self.Phi.dot(self._alpha).dot(self.Phi.T)
        return self._reconstruct(Kmanifold)

    def get_row(self, i):
        approx = self.Phi[i,:].dot(self._alpha).dot(self.Phi.T)
        return self._reconstruct(approx)

    def get_rows(self, rows):
        approx = self.Phi[rows,:].dot(self._alpha).dot(self.Phi.T)
        return self._reconstruct(approx)

    def get_memory(self):
        return (self.Phi.shape[0] * self.Phi.shape[1] + self._alpha.shape[0] * self._alpha.shape[1]) * 8

    def get_name(self):
        return 'sa'

    def set_l(self, l):
        self._l = l
        # reset the solution
        self._alpha = None
        self._inds = None
        self._noninds = None
        self._n = None

    def reset(self, preprocessing=False):
        if preprocessing:
            self.Phi = None
            self.Lambda = None
        self._alpha = None
        self._inds = None
        self._noninds = None


class SMDS(Scaling):
    def __init__(self, k, sa=None):
        super(SMDS, self).__init__(k)
        self._sa = sa

    def preprocessing(self, pts, polys):
        self._sa.preprocessing(pts, polys)

    def fit(self, X):
        n = X.shape[0]
        self._n = n

        # Compute the alpha = MDM^T
        self._sa.fit(X)
        me = self._sa.Phi.shape[1]

        # Apply Spectral MDS
        JPhi = self._sa.Phi - np.ones((n, 1)).dot(np.ones((1, n)).dot(self._sa.Phi)/n)
        u, s, v = np.linalg.svd(JPhi, full_matrices=False)
        svt = np.diag(s).dot(v)
        q = -0.5*svt.dot(self._sa._alpha).dot(svt.T)
        s2, p = la.eigh(q, eigvals=(me - self._k, me - 1))
        Z = u.dot(p.dot(np.diag(np.sqrt(s2))))

        self.Z = Z
        return Z

    def get_memory(self):
        me = self._sa._me
        n = self._n
        return self._sa.get_memory() + (n*me + me*me + me*self._k + self._k) * 8

    def get_name(self):
        return 'smds'
