import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as spl
import scipy.sparse as sps
from sdfs.tpfa import TPFA


class DarcyExp(object):

    def __init__(self, tpfa: TPFA, ssv: npt.NDArray | None = None) -> None:
        self.tpfa = tpfa
        self.ssv = range(self.tpfa.geom.cells.num) if ssv is None else ssv
        self.Nc = self.tpfa.geom.cells.num
        self.Nc_range = np.arange(self.Nc)
        self.cells_neighbors = self.tpfa.cell_neighbors
        self.keep = np.concatenate(
            (self.Nc_range, np.flatnonzero(self.cells_neighbors >= 0) + self.Nc))
        self.cols = np.concatenate(
            (self.Nc_range, np.tile(self.Nc_range, 4)))[self.keep]
        self.rows = np.concatenate(
            (self.Nc_range, self.cells_neighbors.ravel()))[self.keep]
        neumann_bc = (self.tpfa.bc.kind == 'N')
        Nq = np.count_nonzero(neumann_bc)
        self.dLdq = sps.csc_matrix(
            (-np.ones(Nq), (np.arange(Nq), self.tpfa.geom.cells.to_hf[2*self.tpfa.Ni:][neumann_bc])), shape=(Nq, self.Nc))

    def randomize_bc(self, kind: str, scale: float):
        self.tpfa.bc.randomize(kind, scale)
        self.tpfa.update_rhs(kind)
        return self

    def increment_bc(self, kind: str, value: float):
        self.tpfa.bc.increment(kind, value)
        self.tpfa.update_rhs(kind)
        return self

    def solve(self, Y: npt.NDArray, q: npt.NDArray | None = None) -> npt.NDArray:
        self.K = np.exp(Y)
        self.A, b = self.tpfa.ops(self.K, q)
        return spl.spsolve(self.A, b)

    def residual(self, u: npt.NDArray, Y: npt.NDArray) -> npt.NDArray:
        self.K = np.exp(Y)
        self.A, b = self.tpfa.ops(self.K)
        return self.A.dot(u) - b

    def residual_sens_Y(self, u: npt.NDArray, Y: npt.NDArray) -> npt.NDArray:
        # call residual(self, u, Y) before residual_sens_Y(self, u, Y)
        offdiags = (u[self.cells_neighbors] - u[None, :]) * self.tpfa.sens()
        vals = np.vstack(((self.tpfa.alpha_dirichlet * u - self.tpfa.rhs_dirichlet -
                           offdiags.sum(axis=0))[None, :], offdiags)) * self.K[None, :]
        return sps.csr_matrix((vals.ravel()[self.keep], (self.rows, self.cols)), shape=(self.Nc, self.Nc))

    def residual_sens_u(self, u: npt.NDArray, Y: npt.NDArray) -> npt.NDArray:
        # call residual(self, u, Y) before residual_sens_u(self, u, Y)
        return self.A

    def residual_sens_p(self, u: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
        # call residual(self, u, Y) before residual_sens_p(self, u, p)
        return sps.vstack([self.residual_sens_Y(u, p[:self.tpfa.geom.cells.num]), self.dLdq])
