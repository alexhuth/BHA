# BHA
Biharmonic matrix approximation (BHA) is a sparse method for efficiently representing geodesic distance matrices (or other symmetric matrices defined over 3D manifolds) using biharmonic interpolation. These representations can be used to (very efficiently) do multidimensional scaling (MDS) on geodesic distance matrices, a 

<img src="header.png" width="65%">

## Get started
```
pip install -U cython numpy
pip install -r requirements.txt

python bha/test_bha_mesh.py
```
This will install the relevant dependencies (so you might want to do it in a virtual environment!) and run an example.
