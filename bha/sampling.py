import numpy as np


class Sampling(object):
    def __init__(self, func, k_to_all=True):
        self._func = func
        self._k_to_all = k_to_all

    def sample(self, X, k):
        pass


class RandomSampling(Sampling):
    def sample(self, X, k):
        n = X.shape[0]
        # Draw the column indices
        inds = np.random.permutation(n)[:k]
        inds.sort()

        if self._k_to_all:
            C = self._func(X, inds, np.arange(n))
        else:
            C = self._func(X, inds, inds)

        return inds, C


class FarthestSampling(Sampling):
    def sample(self, X, k):
        n = X.shape[0]
        all_idx = np.arange(n)

        if self._k_to_all:
            W = np.empty((n, k))
        else:
            W = np.empty((k, k))

        inds = np.empty((k,), dtype=np.int)

        # Choose initial landmark randomly
        inds[0] = np.random.randint(0, n)
        Fmin = self._func(X, np.array([inds[0]]), all_idx)
        Fmin[inds[0]] = 0.0
        if self._k_to_all:
            W[:, 0] = Fmin.squeeze().copy()
        else:
            W[0, 0] = 0.0

        for i in range(1, k):
            inds[i] = np.argmax(Fmin)
            # print(i, '-', inds[i])
            F = self._func(X, np.array([inds[i]]), all_idx)
            F[inds[i]] = 0.0 # TODO: Fmin itself should be 0 already, right?
            Fmin = np.minimum(Fmin, F)
            if self._k_to_all:
                W[:, i] = F.squeeze()
            else:
                W[:i, i] = F[inds[:i]].squeeze()
                W[i, :i] = F[inds[:i]].squeeze()
                W[i, i] = 0.0

        # sort the indices and permute matrix W
        new_inds = np.argsort(inds)
        if self._k_to_all:
            # sort only columns
            W = W[:, new_inds]
        else:
            # sort columns and rows
            TMP = W[:, new_inds].copy()
            W = TMP[new_inds, :]
        inds = inds[new_inds]

        return inds, W

