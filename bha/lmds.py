import numpy as np
import scipy

from sampling import RandomSampling, FarthestSampling
from mds import Scaling


class LMDS(Scaling):
    """
    Landmark MultiDimensional Scaling

    References:
        [1]_ "Sparse multidimensional scaling using landmark points"; V. de Silva and J. B. Tenenbaum

    """
    def __init__(self, k, l, distFun, applyPCA=True, seed=0, sampling_method='random'):
        super(LMDS, self).__init__(k)
        self._l = l
        self._distance = distFun
        self._applyPCA = applyPCA
        self._seed = seed
        if sampling_method=='random':
            self._sampling_method = RandomSampling(distFun, k_to_all=True)
        elif sampling_method=='farthest':
            self._sampling_method = FarthestSampling(distFun, k_to_all=True)
        else:
            raise ValueError('Unknown sampling method')

    def fit(self, data):
        # Get the number of samples
        n = data.shape[0]

        # Select the landmarks
        landmarks, deltas = self._sampling_method.sample(data, self._l)

        # Apply Classical MDS
        # Compute distance between the landmarks
        landmark_deltas = deltas[landmarks, :]
        landmark_deltas = (landmark_deltas + landmark_deltas.T) / 2.0

        # Compute the means and center around these
        landmark_mu_i = landmark_deltas.mean(axis=1)[:,np.newaxis]
        landmark_mu = landmark_mu_i.mean()
        B = -(landmark_deltas - landmark_mu_i - landmark_mu_i.T + landmark_mu)/2

        w, v = scipy.linalg.eigh(B, eigvals=(self._l-self._k, self._l-1))
        k_final = np.sum(w > 0)
        L = v[:, w > 0].dot(np.diag(np.sqrt(w[w > 0])))

        # Apply distance based triangulation
        L_pinv = v[:, w > 0].dot(np.diag(1/np.sqrt(w[w > 0])))
        X = -L_pinv.T.dot((deltas.T - landmark_mu_i) / 2)

        # Save the resulting model
        self._n = n
        self._k_final = k_final
        self._L = L

        # Apply PCA normalization
        if self._applyPCA:
            self.Z = self._compute_pca(X, k_final).T
        else:
            self.Z = X.T

    def transform(self, data):
        return self.Z

    @staticmethod
    def _compute_pca(X, k):
        X_mean = X.mean(axis=1)
        X_bar = X - X_mean[:,np.newaxis]
        _, U = scipy.linalg.eigh(X_bar.dot(X_bar.T), eigvals=(0, k-1))
        return U.T.dot(X_bar)

    def get_memory(self):
        l = self._l
        n = self._n
        return (n*l + l*l + self._k + self._k*l + l) * 8

    def get_name(self):
        return 'lmds'
