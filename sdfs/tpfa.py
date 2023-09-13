import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from sdfs.geom_mrst import GeomMRST
from sdfs.bc_mrst import BCMRST


class TPFA(object):

    def __init__(self, geom: GeomMRST, bc: BCMRST) -> None:
        self.geom = geom
        self.bc = bc

        self.Nc = self.geom.cells.num
        Nc_range = np.arange(self.Nc)

        self.Ni = self.geom.faces.num_interior
        self.neighbors, self.boundary = np.array_split(
            self.geom.cells.to_hf, [2*self.Ni])
        self.rows = np.concatenate((self.neighbors, Nc_range))
        self.cols = np.concatenate(
            (np.roll(self.neighbors, self.Ni), Nc_range))

        c = self.geom.faces.centroids[:, self.geom.faces.to_hf] - \
            self.geom.cells.centroids[:, self.geom.cells.to_hf]
        n = self.geom.faces.normals[:, self.geom.faces.to_hf]
        n[:, self.Ni:2*self.Ni] *= -1
        self.alpha_interior, self.alpha_boundary = np.array_split(
            np.sum(c * n, axis=0) / np.sum(c ** 2, axis=0), [2*self.Ni])

        self.cell_hfs = np.ascontiguousarray(np.argsort(
            self.geom.cells.to_hf).reshape(4, -1, order='F'))
        self.cell_ihfs = np.where(self.cell_hfs < 2*self.Ni, self.cell_hfs, -1)
        self.cell_neighbors = np.where(self.cell_ihfs >= 0,
                                       self.geom.cells.to_hf[(
                                           self.cell_ihfs + self.Ni) % (2*self.Ni)],
                                       -1)
        self.alpha_dirichlet = np.bincount(self.boundary,
                                           self.alpha_boundary *
                                           (self.bc.kind == 'D'),
                                           minlength=self.Nc)
        self.rhs_dirichlet = np.bincount(self.boundary,
                                         self.alpha_boundary *
                                         (self.bc.kind == 'D') * self.bc.val,
                                         minlength=self.Nc)
        self.rhs_neumann = np.bincount(self.boundary,
                                       (self.bc.kind == 'N') * self.bc.val,
                                       minlength=self.Nc)

    def update_rhs(self, kind: str) -> None:
        if kind == 'D':
            self.rhs_dirichlet = np.bincount(self.boundary,
                                             self.alpha_boundary *
                                             (self.bc.kind == 'D') * self.bc.val,
                                             minlength=self.Nc)
        elif kind == 'N':
            self.rhs_neumann = np.bincount(self.boundary,
                                           (self.bc.kind == 'N') * self.bc.val,
                                           minlength=self.Nc)

    def randomize_bc(self, kind: str, scale: np.float_) -> None:
        self.bc.randomize(kind, scale)
        self.update_rhs(kind)
    
    def increment_bc(self, kind: str, value: np.float_) -> None:
        self.bc.increment(kind, value)
        self.update_rhs(kind)

    def ops(self,
            K: npt.NDArray[np.float_],
            q: npt.NDArray[np.float_] | None = None
           ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        self.K = K
        self.Thf_interior = self.alpha_interior * K[self.geom.cells.to_hf[:2*self.Ni]]
        self.Tgf_interior = (lambda x: x.prod(axis=0) / x.sum(axis=0))(self.Thf_interior.reshape((2, -1)))
        diag = np.bincount(self.neighbors, np.concatenate((self.Tgf_interior, self.Tgf_interior)), minlength=self.Nc) + self.alpha_dirichlet * K
        return sps.csc_matrix((np.concatenate((-self.Tgf_interior, -self.Tgf_interior, diag)), (self.rows, self.cols)), shape=(self.Nc, self.Nc)), \
            self.rhs_dirichlet * K + (self.rhs_neumann if q is None else np.bincount(self.boundary[self.bc.kind == 'N'], q, minlength=self.Nc))

    def sens(self,
             K: npt.NDArray[np.float_]
            ) -> npt.NDArray[np.float_]:
        if self.K is not K:
            print("update K")
            self.K = K
            self.Thf_interior = self.alpha_interior * K[self.geom.cells.to_hf[:2*self.Ni]]
            self.Tgf_interior = (lambda x: x.prod(axis=0) / x.sum(axis=0))(self.Thf_interior.reshape((2, -1)))
        return np.append(self.alpha_interior * ((np.tile(self.Tgf_interior, 2) / self.Thf_interior) ** 2), 0.0)[self.cell_ihfs]
