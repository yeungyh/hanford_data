import numpy as np
import h5py
from geom_mrst import GeomMRST

class BCMRST(object):

    def __init__(self, geom: GeomMRST, filename: str ) -> None:
        num_ext_faces = geom.faces.num - geom.faces.num_interior
        self.kind = np.empty(num_ext_faces, dtype='<U1')
        self.init_val = np.zeros(num_ext_faces)
        
        ext = geom.faces.int_ext[geom.faces.num_interior:]
        ext_idx = np.argsort(ext)
        sorted_ext = ext[ext_idx]
        with h5py.File(filename, 'r') as f:
            idx = ext_idx[np.searchsorted(sorted_ext, f.get('index')[:].ravel() - 1)]
            self.init_val[idx] = f.get('value')[:]
            self.kind[idx] = np.vectorize(chr)(f.get('kind')[:])
        self.val = np.copy(self.init_val)
        self.mean = {k : np.mean(self.init_val[self.kind == k]) for k in np.unique(self.kind)}

    def randomize(self, kind: str, scale: float, rs=np.random.RandomState() ) -> None:
        self.val[self.kind == kind] = rs.normal(self.init_val[self.kind == kind], self.mean[kind] * scale)
    
    def increment(self, kind: str, value: float ) -> None:
        self.val[self.kind == kind] += value
        
    def rescale(self, kind: str, scale: float ) -> None:
        exp_scale = np.exp(scale)
        self.val[self.kind == kind]      /= exp_scale
        self.init_val[self.kind == kind] /= exp_scale
        self.mean[kind]                  /= exp_scale
