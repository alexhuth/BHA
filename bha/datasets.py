import os
import cortex.formats

DATA_DIR = 'data'

def brain_7k():
    pts, polys = cortex.formats.read_obj(os.path.join(DATA_DIR, 'brain_mesh_7k.obj'))
    return pts, polys
