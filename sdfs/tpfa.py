import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from geom_mrst import GeomMRST
from bc_mrst import BCMRST


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
        self.alpha = np.sum(c * n, axis=0) / np.sum(c ** 2, axis=0)

        self.cell_hfs = np.ascontiguousarray(np.argsort(
            self.geom.cells.to_hf).reshape(4, -1, order='F'))
        self.cell_ihfs = np.where(self.cell_hfs < 2*self.Ni, self.cell_hfs, -1)
        self.cell_neighbors = np.where(self.cell_ihfs >= 0,
                                       self.geom.cells.to_hf[(
                                           self.cell_ihfs + self.Ni) % (2*self.Ni)],
                                       -1)
        self.alpha_dirichlet = np.bincount(self.boundary,
                                           self.alpha[2*self.Ni:] *
                                           (self.bc.kind == 'D'),
                                           minlength=self.Nc)
        self.rhs_dirichlet = np.bincount(self.boundary,
                                         self.alpha[2*self.Ni:] *
                                         (self.bc.kind == 'D') * self.bc.val,
                                         minlength=self.Nc)
        self.rhs_neumann = np.bincount(self.boundary,
                                       (self.bc.kind == 'N') * self.bc.val,
                                       minlength=self.Nc)

    def update_rhs(self, kind: str) -> None:
        if kind == 'D':
            self.rhs_dirichlet = np.bincount(self.boundary,
                                             self.alpha[2*self.Ni:] *
                                             (self.bc.kind == 'D') * self.bc.val,
                                             minlength=self.Nc)
        elif kind == 'N':
            self.rhs_neumann = np.bincount(self.boundary,
                                           (self.bc.kind == 'N') * self.bc.val,
                                           minlength=self.Nc)

    def ops(self, K: npt.NDArray, q: npt.NDArray | None = None) -> tuple[npt.NDArray, npt.NDArray]:
        self.Thf_interior = self.alpha[:2*self.Ni] * \
            K[self.geom.cells.to_hf[:2*self.Ni]]
        self.Tgf_interior = (lambda x: x.prod(axis=0) /
                             x.sum(axis=0))(self.Thf_interior.reshape((2, -1)))
        diag = np.bincount(self.neighbors, np.concatenate(
            (self.Tgf_interior, self.Tgf_interior)), minlength=self.Nc) + self.alpha_dirichlet * K
        return sps.csc_matrix((np.concatenate((-self.Tgf_interior, -self.Tgf_interior, diag)), (self.rows, self.cols)), shape=(self.Nc, self.Nc)), \
            self.rhs_dirichlet * K + (np.bincount(self.boundary[self.bc.kind == 'N'], q, minlength=self.Nc) if q is not None else self.rhs_neumann)

    def sens(self) -> npt.NDArray:
        return np.append(self.alpha[:2*self.Ni] * ((np.tile(self.Tgf_interior, 2) / self.Thf_interior) ** 2), 0.0)[self.cell_ihfs]
