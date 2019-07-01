import numpy as np
import os
from cortex.polyutils import Surface

from bha.utils import counter
import cortex


class SymMatrixApprox(object):
    def get_row(self, i):
        return None

    def get_col(self, j):
        return self.get_row(j).T

    def get_elem(self, i, j):
        return self.get_row(i)[j]

    def get_rows(self, rows):
        return None

    def get_size(self):
        return 0

    def score(self, K):
        if self._n != K.get_size():
            print("Warning: wrong matrix sizes for scoring [",self._n,",", K.get_size(),"]")
            return -1.0
        num = 0.0
        denom = 0.0
        for i in range(self._n):
            rowi = self.get_row(i)
            Krowi = K.get_row(i)
            num += np.sum((rowi - Krowi) ** 2)
            denom += np.sum((Krowi) ** 2)
        return num / denom

    def score_rows(self, K, chunk_size=16):
        if self._n != K.get_size():
            print("Warning: wrong matrix sizes for scoring [",self._n,",", K.get_size(),"]")
            return -1.0
        if chunk_size < 0:
            chunk_size = 16
        num = 0.0
        denom = 0.0
        for i in range(0,self._n,chunk_size):
            end_chunk = min(self._n, i+chunk_size)
            rows = self.get_rows(np.arange(i,end_chunk))
            Krows = K.get_rows(np.arange(i,end_chunk))
            num += np.sum((rows - Krows) ** 2)
            denom += np.sum((Krows) ** 2)
        return num / denom

    def score_partial(self, K, total_rows=None, chunk_size=16):
        if self._n != K.get_size():
            print("Warning: wrong matrix sizes for scoring [",self._n,",", K.get_size(),"]")
            return -1.0
        if chunk_size < 0:
            chunk_size = 16
        if total_rows is None or total_rows < 1:
            # If not pointed out, use 10%
            total_rows = (int)(self._n * 10 // 100)

        # Draw total_rows out of n
        partial_rows = np.random.choice(self._n, size=total_rows, replace=False)
        num = 0.0
        denom = 0.0
        for i in range(0, total_rows, chunk_size):
            end_chunk = min(total_rows, i+chunk_size)
            rows = self.get_rows(partial_rows[i:end_chunk])
            Krows = K.get_rows(partial_rows[i:end_chunk])
            num += np.sum((rows - Krows) ** 2)
            denom += np.sum((Krows) ** 2)
        return num / denom

    def score_sel_rows(self, Krows, sel_rows):
        rows = self.get_rows(sel_rows)
        #Krows = K.get_rows(sel_rows)
        num = np.sum((rows - Krows) ** 2, 1)
        denom = np.sum(Krows ** 2, 1)
        return num / denom

    def get_memory(self):
        pass

    def isMDS(self):
        return False

    def get_name(self):
        return 'SymMatrixApprox'

    def reset(self, preprocessing=False):
        pass


class FullK(SymMatrixApprox):
    def __init__(self, func=None):
        self._func = func

    def fit(self, X, y=None):
        if not self._func is None:
            K = self._func(X)
        else:
            K = X.dot(X.T)

        self._n = K.shape[0]
        self._K = K

    def get_row(self, i):
        return self._K[i, :]

    def get_rows(self, rows):
        return self._K[rows, :]

    def get_size(self):
        return self._n

    def get_memory(self):
        return self._K.shape[0] * self._K.shape[1] * 8

    def get_name(self):
        return 'FullK'

    @property
    def shape(self):
        return self._n, self._n

class FullKLazy(SymMatrixApprox):
    def __init__(self, func=None):
        self._func = func

    def fit(self, X, y=None):
        self._n = X.shape[0]
        self._X = X

    def get_row(self, i):
        if not self._func is None:
            row = self._func(self._X, np.asarray(i))
            if row.shape[0] != 1:
                row = row.T
        else:
            row = self._X[i,:].dot(self._X.T)
        return row.ravel()

    def get_rows(self, rows):
        if not self._func is None:
            result = self._func(self._X, rows)
            if result.shape[0] != rows.size:
                result = result.T
        else:
            result = self._X[rows,:].dot(self._X.T)
        return result

    def get_size(self):
        return self._n

    def get_memory(self):
        return self._X.shape[0] * self._X.shape[1] * 8

    def get_name(self):
        return 'FullKLazy'

    @property
    def shape(self):
        return self._n, self._n

class MeshK(SymMatrixApprox):
    CACHE_DIR = "/tmp"
    
    def __init__(self, func=None, m=1.0):
        self._m = m
        if func is not None:
            self._func = func
        else:
            self._func = lambda x: x

    def fit(self, pts, polys, recache=False):
        self._n = len(pts)
        self._surface = Surface(pts, polys)

        # check to see if the full K is cached
        dataset_hash = str(hash(pts.tostring())) + str(hash(polys.tostring())) + str(hash(self._m))
        cache_path = os.path.join(self.CACHE_DIR, dataset_hash+'_geodesic_K_cache.npz')
        if not os.path.exists(cache_path) or recache:
            print('Cache not found, computing K..')
            K = np.vstack([self._surface.geodesic_distance([i], m=self._m) for i in counter(range(self._n))])
            self._K = (K + K.T) / 2.0
            np.savez(cache_path, geodesic_K=self._K)
        else:
            print('Loading K from cache..')
            self._K = np.load(cache_path)['geodesic_K']
    
    def get_row(self, i):
        return self._K[i,:]

    def get_rows(self, rows):
        return self._K[rows,:]

    def get_size(self):
        return self._n

    def get_memory(self):
        return self._K.shape[0] * self._K.shape[1] * 8

    def get_name(self):
        return 'MeshK'

    @property
    def shape(self):
        return self._n, self._n

class MeshKLazy(SymMatrixApprox):
    def __init__(self, func=None, m=1.0):
        """
        m : float, default 1.0
            Time step for the geodesic heat approximation. Should be 1.0 normally,
            but if there's numerical instability, set it higher (e.g. 100.0)
        """
        if func is not None:
            self._func = func
        else:
            self._func = lambda x: x

        self._m = m
        
    def fit(self, pts, polys):
        self._n = len(pts)
        self._surface = Surface(pts, polys)
    
    def get_row(self, i):
        return self._func(self._surface.geodesic_distance([i], m=self._m))

    def get_rows(self, rows):
        return np.vstack([self._func(self._surface.geodesic_distance([i], m=self._m)) for i in rows])

    def get_size(self):
        return self._n

    def get_memory(self):
        return (self._surface.pts.shape[0] *  self._surface.pts.shape[1] + self._surface.polys.shape[0] * self._surface.polys.shape[1]) * 8

    def get_name(self):
        return 'MeshKLazy'

    @property
    def shape(self):
        return self._n, self._n


class RankK(SymMatrixApprox):
    def __init__(self, k, func=None):
        self._func = func
        self._k = k

    def fit(self, X, y=None):
        if not self._func is None:
            K = self._func(X)
        else:
            K = X

        U, S, V = np.linalg.svd(K, full_matrices=False)

        self._n = K.shape[0]
        self._U = U
        self._S = S
        self._V = V

    def get_row(self, i):
        return (self._U[i][:,:self._k] * self._S[:self._k]).dot(self._V[:self._k,:])

    def get_rows(self, rows):
        return (self._U[rows][:,:self._k] * self._S[:self._k]).dot(self._V[:self._k,:])

    def get_size(self):
        return self._n

    def set_k(self, k):
        if k < 1:
            k = 10
            print('K is too low, setting to ',k)
        self._k = k

    def get_memory(self):
        return (self._U.shape[0] * self._U.shape[1] + self._S.shape[0] + 
                self._V.shape[0] * self._V.shape[1] ) * 8

    def get_name(self):
        return 'BestRankK'

