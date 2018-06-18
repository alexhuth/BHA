#from matplotlib.pyplot import *
from bha import BHA
from fmds import MF
from smds import SA
from time import time
from base import (MeshK, MeshKLazy)
from nystrom import Nystrom
import datasets


dataset_name = "brain_7k"

t = 10
l = 50
l_nys = 20
m_eig = 50

chunk_size = 2048

# Brain 7k
pts, polys = getattr(datasets, dataset_name)()


n = len(pts)
X = pts
print(X.shape)

# Create the full K
if n <= 15000:
    # use MeshK (which automatically caches now)
    full = MeshK()
    full.fit(pts, polys)
else:
    # use MeshKLazy
    full = MeshKLazy()
    full.fit(pts, polys)

# Create the metric function
def geodesic(_, source=None, dest=None):
    if source is None and dest is None:
        return np.vstack([full.get_row(i) for i in range(len(pts))])
    elif dest is None:
        return full.get_rows(source).T
    else:
        return full.get_rows(source)[:,dest].T

metric_func = geodesic


## END GEODESICS SETUP

t0 = time()
nys = Nystrom(l_nys, func=metric_func)
nys.fit(X)
t1 = time()
print("-------------------------------")
print("Done unreg Nystrom: rel. error %f " % nys.score_rows(full, chunk_size=chunk_size))
print("Timings:")
print("Fit: %f" % (t1-t0))

t0 = time()
nys = Nystrom(l_nys, func=metric_func, rcond=1e-4)
nys.fit(X)
t1 = time()
print("-------------------------------")
print("Done reg Nystrom: rel. error %f " % nys.score_rows(full, chunk_size=chunk_size))
print("Timings:")
print("Fit: %f" % (t1-t0))

# Compute the MF approximation
t0 = time()
mf = MF(l=l, func=metric_func, soft_p=100000.)
mf.preprocessing(pts, polys)
t1 = time()
mf.fit(X)
t2 = time()
Kmanifold = mf.transform()
t3 = time()
print("-------------------------------")
print("Done MF: rel. error %f " % mf.score_rows(full, chunk_size=chunk_size))
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute the SA approximation
t0 = time()
sa = SA(l, m_eig, func=metric_func)
sa.preprocessing(pts, polys)
t1 = time()
sa.fit(X)
t2 = time()
Kmanifold = sa.transform()
t3 = time()
print("-------------------------------")
print("Done SA: rel. error %f " % sa.score_rows(full, chunk_size=chunk_size))
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute the PWP' approximation
t0 = time()
bha = BHA(l=l, func=metric_func)
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done BHA: rel. error %f " % bha.score_rows(full, chunk_size=chunk_size))
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute the PWP' approximation with farthest point sampling
t0 = time()
bha = BHA(l=l, func=metric_func, sampling_method='farthest')
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done FarthestPS-BHA: rel. error %f " % bha.score(full))
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute the PWP' approximation w/ squared distances
t0 = time()
bha = BHA(l=l, func=metric_func, interp_sq=True)
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done BHA-sq: rel. error %f " % bha.score_rows(full, chunk_size=chunk_size))
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute the PWP' approximation with farthest point sampling & squared dists
t0 = time()
bha = BHA(l=l, func=metric_func, sampling_method='farthest', interp_sq=True)
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done FarthestPS-BHA-sq: rel. error %f " % bha.score(full))
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute the PWP' approximation with FPS, SqD, & soft P
t0 = time()
bha = BHA(l=l, func=metric_func, sampling_method='farthest', interp_sq=True, soft_p=5.)
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done FarthestPS-BHA-sq-soft: rel. error %f " % bha.score(full))
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute PWP' approximation with multiprocessing
t0 = time()
bha = BHA(l=l, func=metric_func)
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done BHA-mp: rel. error %f " % bha.score(full))
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute the thresholding sparse PWP' approximation
t0 = time()
bha = BHA(l=l, nnz_row=l_nys, func=metric_func)
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done l0 Sparse BHA: rel. error %f " % bha.score(full))
print("NNZ P: ", bha._nnz / bha.get_size())
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))


# Compute the thresholding sparse PWP' approximation w/ farthest point sampling
t0 = time()
bha = BHA(l=l, nnz_row=l_nys, func=metric_func, sampling_method='farthest')
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done l0 Sparse FS-BHA: rel. error %f " % bha.score(full))
print("NNZ P: ", bha._nnz / bha.get_size())
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))

# Compute the row-thresholding sparse PWP' approximation
t0 = time()
bha = BHA(l=l, nnz_row=l_nys, sparse_direction='rows', func=metric_func)
bha.preprocessing(pts, polys)
t1 = time()
bha.fit(X)
t2 = time()
Kmanifold = bha.transform()
t3 = time()
print("-------------------------------")
print("Done l0-rows Sparse BHA: rel. error %f " % bha.score(full))
print("NNZ P: ", bha._nnz / bha.get_size())
print("Timings:")
print("Preprocessing: %f" % (t1-t0))
print("Fit: %f" % (t2-t1))
print("Transform: %f" % (t3-t2))
