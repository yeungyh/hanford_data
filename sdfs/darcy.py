import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as spl
import scipy.sparse as sps
from sdfs.tpfa import TPFA


class DarcyExp(object):

    def __init__(self,
                 tpfa: TPFA,
                 iuobs: npt.NDArray[np.int_],
                 ssv: npt.NDArray[np.int_] | None = None
                ) -> None:
        self.tpfa = tpfa
        self.Nc = self.tpfa.geom.cells.num
        self.Nc_range = np.arange(self.Nc)
        self.ssv = range(self.Nc) if ssv is None else ssv
        self.cells_neighbors = self.tpfa.cell_neighbors
        self.keep = np.concatenate(
            (self.Nc_range, np.flatnonzero(self.cells_neighbors >= 0) + self.Nc))
        self.cols = np.concatenate(
            (self.Nc_range, np.tile(self.Nc_range, 4)))[self.keep]
        self.rows = np.concatenate(
            (self.Nc_range, self.cells_neighbors.ravel()))[self.keep]
        neumann_bc = (self.tpfa.bc.kind == 'N')
        Nq = np.count_nonzero(neumann_bc)
        self.dLdq = sps.csc_matrix((-np.ones(Nq),
                                    (self.tpfa.geom.cells.to_hf[2*self.tpfa.Ni:][neumann_bc], np.arange(Nq))),
                                   shape=(self.Nc, Nq))
        self.Y = None

    def randomize_bc(self, kind: str, scale: np.float_):
        self.tpfa.randomize_bc(kind, scale)
        return self

    def increment_bc(self, kind: str, value: np.float_):
        self.tpfa.increment_bc(kind, value)
        return self

    def check_and_update(self,
                         Y: npt.NDArray[np.float_],
                         q: npt.NDArray[np.float_] | None = None
                        ) -> None:
        if self.Y is not Y:
            self.Y, self.K = Y, np.exp(Y)
            self.A, self.b = self.tpfa.ops(self.K, q)

    def solve(self,
              Y: npt.NDArray[np.float_],
              q: npt.NDArray[np.float_] | None = None
             ) -> npt.NDArray[np.float_]:
        self.check_and_update(Y, q)
        return spl.spsolve(self.A, self.b)
    
    def residual(self,
                 u: npt.NDArray[np.float_],
                 Y: npt.NDArray[np.float_],
                 q: npt.NDArray[np.float_] | None = None
                ) -> npt.NDArray[np.float_]:
        self.check_and_update(Y, q)
        return self.A.dot(u) - self.b

    def residual_sens_Y(self,
                        u: npt.NDArray[np.float_],
                        Y: npt.NDArray[np.float_],
                        q: npt.NDArray[np.float_] | None = None
                       ) -> npt.NDArray[np.float_]:
        self.check_and_update(Y, q)
        offdiags = (u[self.cells_neighbors] - u[None]) * self.tpfa.sens(self.K)
        vals = np.vstack(((self.tpfa.alpha_dirichlet * u - self.tpfa.rhs_dirichlet -
                           offdiags.sum(axis=0))[None], offdiags)) * self.K[None]
        return sps.csr_matrix((vals.ravel()[self.keep],
                               (self.rows, self.cols)),
                              shape=(self.Nc, self.Nc))
    
    def residual_sens_p(self,
                        u: npt.NDArray[np.float_],
                        p: npt.NDArray[np.float_]
                       ) -> npt.NDArray[np.float_]:
        return sps.hstack([self.residual_sens_Y(u, *np.array_split(p, [self.Nc])), self.dLdq])

    def residual_sens_u(self,
                        u: npt.NDArray[np.float_],
                        Y: npt.NDArray[np.float_],
                        q: npt.NDArray[np.float_] | None = None
                       ) -> npt.NDArray[np.float_]:
        self.check_and_update(Y, q)
        return self.A