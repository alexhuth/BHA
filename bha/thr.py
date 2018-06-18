import numpy as np
import heapq


class THR(object):
    def __init__(self, k):
        self._k = k

    def fit(self, X, y=None):
        n = X.shape[0]
        sorted_idxs = np.argsort(np.abs(X), axis=0)
        self._idxs = sorted_idxs[n-self._k::]
        self._vals = X[self._idxs]
        return self

    def fit_partition(self, X, y=None):
        n = X.shape[0]
        sorted_idxs = np.argpartition(np.abs(X), n-self._k, axis=0)
        self._idxs = sorted_idxs[n-self._k::]
        self._vals = X[self._idxs]
        return self

    def fit_group(self, X):
        n, m = X.shape
        sorted_idxs = np.argpartition(np.abs(X), n-self._k, axis=0)
        self._idxs = sorted_idxs[n-self._k::,:]
        self._vals = np.empty(m*self._k)
        for i in range(m):
            self._vals[i*self._k:(i+1)*self._k] = X[self._idxs[:,i],i].ravel()
        self._idxs = self._idxs.T.ravel()
        return self


class THR_ROWS(object):
    def __init__(self, k):
        self._k = k
        self._reset = True

    def fit(self, X, y=None):
        n, kx = X.shape


        if self._reset:
            self._reset = False
            # First call to fit(), setup the data structures
            self._n = n
            self._total_k = 0
            self._stored_k = 0
            self._queues = [[] for _ in range(n)]
            # self._row_vals = np.empty((n, self._k))
            # self._row_idxs = np.empty((n, self._k))
        elif self._n != n:
            print("Error fitting two different number of features between calls to fit")
            exit(0)


        first_col = 0
        total_cols = 0
        last_col = kx

        if self._stored_k < self._k:
            # We don't have enough nnz yet, let's add in order until we get k or the next chunk arrives.
            if self._stored_k + kx < self._k:
                # Just add the elements to each row
                total_cols = kx
                first_col = kx
            else:
                total_cols = min(kx, self._k - self._stored_k)
                first_col = total_cols

            for row in range(n):
                for col in range(total_cols):
                    self._queues[row].append([abs(X[row, col]), X[row, col], (col + self._total_k)])

                # heapify if row has k elements
                if self._stored_k + kx >= self._k:
                    heapq.heapify(self._queues[row])

        if first_col < kx:
            # Still have elements to compare and may change the queues
            for row in range(n):
                for col in range(first_col, last_col):
                    # find for the elements that are bigger than thos in the heap and add them to it after removing the
                    # smallest (top) in it.
                    if self._queues[row][0][0] < abs(X[row, col]):
                        heapq.heappop(self._queues[row])
                        heapq.heappush(self._queues[row], [abs(X[row, col]), X[row, col], (col+self._total_k)])

        self._total_k += kx
        self._stored_k += total_cols

        return self

    def get_best_k(self):
        values = np.empty(self._k*self._n)
        columns = np.empty(self._k*self._n, dtype=int)
        idx = 0
        for row in range(self._n):
            while self._queues[row]:
                _, values[idx], columns[idx] = heapq.heappop(self._queues[row])
                # print(values[idx], ' ', columns[idx])
                idx += 1

        return columns, values