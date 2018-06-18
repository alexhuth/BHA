from bha import BHA
from bha import datasets

pts, polys = datasets.brain_7k()
bha = BHA.from_surface(pts, polys, l=500)
