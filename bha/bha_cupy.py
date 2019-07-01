import numpy as np
from scipy import sparse
import scipy.linalg as la
from cortex.polyutils import Surface
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as sparsela
import cupy

from bha.thr import THR, THR_ROWS
from bha.base import SymMatrixApprox, MeshKLazy
from bha.sampling import RandomSampling, FarthestSampling
from bha.mds import Scaling
from bha.utils import counter


class BHA(SymMatrixApprox):
    def __init__(self, l, func=None, nnz_row=None, threshold=None, 
                 sparse_direction='cols', fit_group=True, 
                 sampling_method='farthest', interp_sq=False, soft_p=None):
        self._l = l
        self._func = func
        self._nnz_row = nnz_row
        self._threshold = threshold
        self._sparse_direction = sparse_direction
        self._fit_group = fit_group
        self.set_sampling_method(sampling_method)
        self._interp_sq = interp_sq # interpolate squared distance, then sqrt?
        self._soft_p = soft_p

    def set_sampling_method(self, sampling_method):
        if sampling_method=='random':
            self._sampling_method = RandomSampling(self._func, k_to_all=False)
        elif sampling_method=='farthest':
            self._sampling_method = FarthestSampling(self._func, k_to_all=False)
        else:
            raise ValueError('Unknown sampling method')

    def preprocessing(self, pts, polys):
        self._pts = pts
        self._polys = polys
        self._surface = Surface(pts, polys)

        B, D, lapW, lapV = self._surface.laplace_operator
        npt = len(D)
        Dinv = sparse.dia_matrix((D ** -1, [0]), (npt, npt)).tocsr()  # construct Dinv

        self._M = (lapV - lapW).dot(Dinv.dot(lapV - lapW))

        self._lapW = lapW
        self._lapV = lapV
        self._Dinv = Dinv

    def get_W(self, X, Winds):
        W = self._func(X, Winds, Winds)

        if self._interp_sq:
            return W ** 2
        else:
            return W

    def fit(self, X):
        #(X, Winds, K, t):
        n, d = X.shape
        self._n = n

        # If not computed, then compute the Nearest Neighbor graph
        # and the Laplace-Beltrami Operator.
        # if self._M is None:
        #     self.preprocessing(X)
        if "_Winds" not in dir(self) or self._Winds is None:
            Winds, W = self._sampling_method.sample(X, self._l)
            nonWinds = np.setdiff1d(np.arange(n), Winds)  # non-landmark points
            self._Winds = Winds
            self._nonWinds = nonWinds
            # square the distance matrix if we need to do that
            if self._interp_sq:
                W = W ** 2
        else:
            Winds = self._Winds
            nonWinds = self._nonWinds
            # compute kernel, W
            W = self.get_W(X, Winds)

        self._W = cupy.array((W + W.T) / 2.0)
        
        # compute P
        self.compute_P()

    def compute_P(self):
        Winds = self._Winds
        nonWinds = self._nonWinds
        n = self._M.shape[0]
        
        # pull out part of M for unselected points
        M_aa = self._M[nonWinds,:][:,nonWinds].tocsc()
        
        # pull out part of M that crosses selected and unselected points
        M_ab = self._M[Winds,:][:,nonWinds]

        if self._nnz_row is not None:
            if self._sparse_direction == 'cols':
                self._threshold = self._nnz_row * (n-self._l) // self._l
            else:
                self._threshold = self._nnz_row
        
        try:
            from sksparse.cholmod import cholesky
            solve_method = 'cholmod'
        except ImportError:
            solve_method = 'spsolve'

        # compute Pprime, part of the dense interpolation matrix
        if self._threshold is None:
            if solve_method == 'spsolve':
                Pprime = sparse.linalg.spsolve(M_aa, -M_ab.T)
            elif solve_method == 'cholmod':
                Pprime = cholesky(M_aa).solve_A(-M_ab.T)
        
            # compute P, the full dense interpolation matrix
            P = np.zeros((n, self._l))
            P[nonWinds,:] = Pprime.todense()
            P[Winds,:] = np.eye(self._l)
            Pnnz = n * self._l

            if self._soft_p is not None:
                # don't force P to be exactly identity for known points,
                # allow it to fudge a little
                print("Softening P..")
                M_bb = self._M[Winds,:][:,Winds]
                soft_eye = sparse.eye(self._l) * self._soft_p
                to_invert = (M_bb + soft_eye + M_ab.dot(Pprime)).todense()
                soft_factor = np.linalg.inv(to_invert) * self._soft_p
                P = P.dot(soft_factor).A

        else:
            # Compute the sparse bha
            if solve_method == 'cholmod':
                chol_M_aa = cholesky(M_aa)

            if self._sparse_direction == 'rows':
                thresh = THR_ROWS(k=self._threshold)
                Prows = np.empty(self._threshold*(n-self._l)+self._l, dtype=int)
                Pcols = np.empty(self._threshold*(n-self._l)+self._l, dtype=int)
                Pvals = np.empty(self._threshold*(n-self._l)+self._l)
            else:
                thresh = THR(k=self._threshold)
                Prows = np.empty(self._threshold*self._l+self._l, dtype=int)
                Pcols = np.empty(self._threshold*self._l+self._l, dtype=int)
                Pvals = np.empty(self._threshold*self._l+self._l)
            chunk_size = 64  # min(self._l // self._njobs, 64)
            chunks = self._l // chunk_size + ((self._l % chunk_size) > 0)

            for chunk in counter(range(chunks)):
                start = chunk*chunk_size
                end = min(((chunk+1)*chunk_size, self._l))
                if solve_method == 'spsolve':
                    sol = sparse.linalg.spsolve(M_aa, -M_ab.T[:, start:end].toarray())
                elif solve_method == 'cholmod':
                    sol = chol_M_aa.solve_A(-M_ab.T[:, start:end].toarray())

                if self._sparse_direction == 'rows':
                    thresh.fit(sol)
                else:
                    if self._fit_group:
                        l_i = 0
                        for l in range(start,end):
                            thresh.fit_partition(sol[:, l_i])
                            Prows[l*self._threshold:(l+1)*self._threshold] = nonWinds[thresh._idxs]
                            Pvals[l*self._threshold:(l+1)*self._threshold] = thresh._vals
                            l_i += 1
                    else:
                        l_i = 0
                        for l in range(start,end):
                            thresh.fit(sol[:, l_i])
                            Prows[l*self._threshold:(l+1)*self._threshold] = nonWinds[thresh._idxs]
                            Pvals[l*self._threshold:(l+1)*self._threshold] = thresh._vals
                            l_i += 1

            if self._sparse_direction == 'rows':
                cols, vals = thresh.get_best_k()
                Prows[:(n-self._l)*self._threshold] = np.repeat(nonWinds[np.arange(n-self._l)],self._threshold)
                Pcols[:(n-self._l)*self._threshold] = cols
                Pvals[:(n-self._l)*self._threshold] = vals
                lastnonWindElement = (n-self._l)*self._threshold
            else:
                Pcols[:self._l*self._threshold] = np.repeat(np.arange(self._l),self._threshold)
                lastnonWindElement = self._l*self._threshold

            # add the identity for indices in W
            Prows[lastnonWindElement:] = Winds
            Pcols[lastnonWindElement:] = np.arange(self._l)
            Pvals[lastnonWindElement:] = 1.0

            P = sparse.csr_matrix((Pvals,(Prows, Pcols)), shape=(n,self._l))
            P.eliminate_zeros()
            Pnnz = P.nnz

        # save values
        self._nnz = Pnnz
        self._P = cupy.sparse.csr_matrix(P)

    def reconstruct(self, approx):
        """Reconstruct the data from the approximation. Takes the square root
        of the approximation if we are approximating the squared matrix.
        """
        if self._interp_sq:
            return np.sqrt(np.clip(approx, 0, np.inf))
        else:
            return approx
    
    def transform(self):
        if self._threshold is None:
            Kmanifold = self._P.dot(self._W).dot(self._P.T)
        else:
            Kmanifold = self._P.dot(self._P.dot(self._W).T)

        return self.reconstruct(Kmanifold)

    def get_row(self, i, out=None):
        if self._threshold is None:
            approx = self._P[i,:].dot(self._W).dot(self._P.T)
        else:
            if not hasattr(self, '_cupyPW'):
                # create GPU objects to hold intermediate states so we don't re-create them every time
                self._cupyPW = cupy.zeros((1, self._W.shape[0]))
                self._cupyrow = cupy.zeros((self._P.shape[0],))
                self._P.sum_duplicates()

            # extract the row of P, make sure it has the right format (!)
            Pi = self._P[i]
            #Pi.sum_duplicates()
            Pi._has_canonical_format = True # since P is already in canonical format?

            cupy.cusparse.csrmm2(Pi, self._W.T, self._cupyPW)
            if out is not None:
                cupy.cusparse.csrmv(self._P, self._cupyPW.T, out)
                return out
            else:
                cupy.cusparse.csrmv(self._P, self._cupyPW.T, self._cupyrow)
                #approx = self._P.dot(self._P[i,:].dot(self._W).T).T
                approx = self._cupyrow.get()

        return self.reconstruct(approx)

    def get_rows(self, rows):
        raise NotImplementedError('cupy version doesn\'t support get_rows, sorry')

        if self._threshold is None:
            approx = self._P[rows,:].dot(self._W).dot(self._P.T)
        else:
            approx = self._P.dot(self._P[rows,:].dot(self._W).T).T

        return self.reconstruct(approx)

    def get_size(self):
        return self._n

    def set_l(self, l):
        self._l = l
        # reset the solution
        self._nonWinds = None
        self._Winds = None
        self._threshold = None
        self._n = None
        self._nnz = 0
        self._P = None
        self._W = None

    def set_nnz_row(self, nnz_row=None):
        self._nnz_row = nnz_row
        # reset the solution
        self._threshold = None
        self._n = None
        self._nnz = 0
        self._P = None
        self._W = None

    def get_memory(self):
        if self._nnz_row == None:
            return (self._W.shape[0] * self._W.shape[1] + 
                    self._P.shape[0] * self._P.shape[1]) * 8
        else:
            return (self._W.shape[0] * self._W.shape[1] + self._P.data.shape[0] + 
                    self._P.indices.shape[0]//2 + self._P.indptr.shape[0]//2) * 8

    def get_name(self):
        if self._nnz_row == None:
            return 'bha'
        else:
            return 'sbha' + str(self._nnz_row)

    def reset(self, preprocessing=False):
        # reset the solution
        if preprocessing:
            self._M = None
            self._lapW = None
            self._lapV = None
            self._Dinv = None
        self._nonWinds = None
        self._Winds = None

    @classmethod
    def from_surface(cls, pts, polys, l, nnz_row=150, m=1.0, **kwargs):

        # create MeshKLazy object that computes geodesics
        meshk = MeshKLazy(m=m)
        meshk.fit(pts, polys)

        # create function that we'll pass to BHA object
        def geodesic(_, source=None, dest=None):
            if source is None and dest is None:
                return np.vstack([meshk.get_row(i) for i in range(len(pts))])
            elif dest is None:
                return meshk.get_rows(source).T
            else:
                return meshk.get_rows(source)[:,dest].T

        # create BHA object
        bha = cls(l, geodesic, nnz_row=nnz_row, **kwargs)

        # preprocess & fit
        bha.preprocessing(pts, polys)
        bha.fit(meshk)

        return bha
    
    def save(self, filename):
        np.savez(filename, W=cupy.asnumpy(self._W), P=cupy.asnumpy(self._P), interp_sq=cupy.asnumpy(self._interp_sq), threshold=self._threshold)

    @classmethod
    def load(cls, filename):
        new_bha = cls.__new__(cls)
        data = np.load(filename)
        new_bha._W = cupy.array(data['W'])
        new_bha._P = cupy.sparse.csr_matrix(data['P'].tolist())
        new_bha._interp_sq = data['interp_sq']
        new_bha._threshold = data['threshold']
        return new_bha


class BMDS(Scaling):
    def __init__(self, k, bha=None, method='lanczos'):
        """
        
        :param k: the embedding dimension 
        :param bha: a BHA method preprocessed
        :param method: 'lanczos' or 'qr'
        """
        super(BMDS, self).__init__(k)
        self._bha = bha
        self._method = method

    def preprocessing(self, pts, polys):
        self._bha.preprocessing(pts, polys)

    def fit(self, X):
        self._n = X.shape[0]
        # 1. Compute W and P matrices
        self._bha.fit(X)

        # 2. Apply the method and obtain Z
        if self._method == 'lanczos':
            self.Z = self._lanczos(X)
        elif self._method == 'qr':
            self.Z = self._qr(X)
        elif self._method == 'ransvd':
            raise ValueError('Method not implemented: %s' % self._method)
        else:
            raise ValueError(self._method)

        return self.Z

    def _lanczos(self, X):
        P = self._bha._P
        W = self._bha._W
        n = X.shape[0]
        PcolsT = P.T.dot(np.ones((n, 1)))/n
        Ones = np.ones((1, n))

        # 1. Define the linear operator for JPWP^TJ
        def lin_op(v):
            x = W.dot(P.T.dot(v) - PcolsT.dot(Ones.dot(v)))
            y = -0.5*(P.dot(x) - Ones.T.dot(PcolsT.T.dot(x)))
            return y
        A = LinearOperator((n, n), matvec=lin_op)

        # 2. Truncated Eigenvalues of Y = V D V^T
        D, V = sparsela.eigsh(A, k=self._k, which='LA')
        Z = V[:, :self._k].dot(np.diag(np.sqrt(D[:self._k])))

        return Z

    def _qr(self, X):
        P = self._bha._P
        W = self._bha._W
        n = X.shape[0]
        l = W.shape[0]

        # 1. Compute A=JP
        A = P - P.T.dot(np.ones((n, 1))).dot(np.ones((1, n))/n).T

        # 2. Compute A=QR
        Q, R = np.linalg.qr(A, mode='reduced')

        # 3. Compute Y=-1/2 RWR^T
        Y = -0.5*(R.dot(W).dot(R.T))

        # 4. Truncated Eigendecomposition of Y=V D V^T with k eigenvalues
        D, V = la.eigh(Y, eigvals=(l - self._k, l - 1))

        # 5. Compute Z = Q V D^1/2
        Z = Q.dot(V.dot(np.diag(np.sqrt(D))))

        return Z

    def get_memory(self):
        l = self._bha._P.shape[1]
        n = self._n
        if self._method == 'lanczos':
            return self._bha.get_memory() + (l + n*self._k + self._k) * 8
        elif self._method == 'qr':
            return self._bha.get_memory() + (n*l + l*l + l*self._k + self._k) * 8

    def get_name(self):
        if self._bha._nnz_row == None:
            return 'bhamds' + self._method
        else:
            return 'sbhamds' + self._method + str(self._bha._nnz_row)
