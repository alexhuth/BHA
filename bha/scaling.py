import os
import numpy as np


class Scaling(object):
    def __init__(self, k):
        self._k = k

    def preprocessing(self, pts, polys):
        pass

    def fit(self, X):
        self._n = X.shape[0]
        self.Z = None

    def stress(self, JEJ):
        ZZT = self.Z.dot(self.Z.T)
        total_stress = np.linalg.norm(ZZT + 0.5 * JEJ, ord='fro') / (float)(self._n * self._n)
        return total_stress

    def get_embedding_rows(self, rows):
        pass

    def stress_partial(self, JEJ, total_rows=None, chunk_size=16):
        # if self._n != JEJ.get_size():
        #     print("Warning: wrong matrix sizes for scoring [",self._n,",", K.get_size(),"]")
        #     return -1.0
        if chunk_size < 0:
            chunk_size = 16
        if total_rows is None or total_rows < 1:
            # If not pointed out, use 10%
            total_rows = (int)(self._n * 10 // 100)

        # Draw total_rows out of n
        partial_rows = np.random.choice(self._n, size=total_rows, replace=False)
        partial_stress = 0.0
        for i in range(0, total_rows, chunk_size):
            end_chunk = min(total_rows, i+chunk_size)
            rows = self.get_embedding_rows(partial_rows[i:end_chunk])
            JEJrows = 0.5 * JEJ.get_rows(partial_rows[i:end_chunk])
            partial_stress += np.sum((rows + JEJrows) ** 2)
        return np.sqrt(partial_stress) / (float)(total_rows * self._n)

    def get_memory(self):
        pass

    def isMDS(self):
        return True

    def get_name(self):
        return 'scaling'


class MDS(Scaling):
    CACHE_DIR = "/tmp"

    def __init__(self, k, func=None, cache=False):
        super(MDS, self).__init__(k)
        self._func=func
        self._cache = cache

    def preprocessing(self, pts, polys):
        if self._cache:
            # check to see if the full K is cached
            dataset_hash = str(hash(pts.tostring())) + str(hash(polys.tostring()))
            self._cache_path = os.path.join(self.CACHE_DIR, dataset_hash+'_mds_cache.npz')

    def fit(self, X):
        n = X.shape[0]
        self._n = n

        if not self._cache:
            self.Z = self._compute_mds(X)
        elif not os.path.exists(self._cache_path):
            print('Cache not found, computing MDS..')
            self.Z = self._compute_mds(X)
            np.savez(self._cache_path, Z=self.Z)
        else:
            print('Loading MDS from cache..')
            self.Z = np.load(self._cache_path)['Z']

        return self.Z

    def _compute_mds(self, X):
        n = X.shape[0]
        all_n = np.arange(n)
        E = self._func(X, all_n, all_n)
        E = (E + E.T)/2
        J = np.eye(n) - np.ones((n, 1)).dot(np.ones((1, n))) / n
        A = -0.5 * (J.dot(E).dot(J))
        W, V = la.eigh(A, eigvals=(n - self._k, n - 1))
        Z = V.dot(np.diag(np.sqrt(W)))
        return Z

    def get_embedding_rows(self, rows):
        return self.Z[rows,:].dot(self.Z.T)

    def get_memory(self):
        return self._n * self._n * 8

    def get_name(self):
        return 'mds'
