import numpy as np
import numpy.typing as npt
import h5py
from dataclasses import dataclass
from collections import namedtuple
from matplotlib import patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.axes import Axes

Nodes = namedtuple('Nodes', ['num', 'coords'])
Cells = namedtuple('Cells', ['num', 'centroids', 'nodes', 'to_hf', 'volumes'])

@dataclass
class Nodes:
    num: int
    coords: npt.NDArray[np.float_]

@dataclass
class Cells:
    num: int
    nodes: npt.NDArray[np.int_]
    centroids: npt.NDArray[np.float_]
    to_hf: npt.NDArray[np.int_]
    volumes: npt.NDArray[np.float_]

@dataclass
class Faces:
    num: int
    nodes: npt.NDArray[np.int_]
    centroids: npt.NDArray[np.float_]
    to_hf: npt.NDArray[np.int_]
    areas: npt.NDArray[np.float_]
    normals: npt.NDArray[np.float_]
    neighbors: npt.NDArray[np.int_]
    num_interior: int
    int_ext: npt.NDArray[np.int_]

class GeomMRST:

    def __init__(self, filename: str) -> None:

        with h5py.File(filename, 'r') as f:
            faces_num = int(f.get('faces/num')[:].item())
            faces_neighbors = f.get('faces/neighbors')[:].astype(int) - 1

            is_interior = np.logical_and(*(faces_neighbors >= 0))
            faces_num_interior = np.count_nonzero(is_interior)
            faces_int_ext = np.argsort(~is_interior)
            Ni_range = np.arange(faces_num_interior)
            faces_to_hf = np.concatenate(
                (Ni_range, Ni_range, np.arange(faces_num_interior, faces_num)))

            faces_neighbors = faces_neighbors[:, faces_int_ext]
            faces_nodes = (f.get(
                'faces/nodes')[:].astype(int) - 1).reshape((2, -1), order='F')[:, faces_int_ext]
            faces_centroids = f.get(
                'faces/centroids')[:][:, faces_int_ext]
            faces_areas = f.get(
                'faces/areas')[:].ravel()[faces_int_ext]
            faces_normals = f.get(
                'faces/normals')[:][:, faces_int_ext]
            faces_normals[:, faces_num_interior:] *= np.array(
                [1, -1]).dot(faces_neighbors[:, faces_num_interior:] >= 0)
            
            self.faces = Faces(faces_num,
                               faces_nodes,
                               faces_centroids,
                               faces_to_hf,
                               faces_areas,
                               faces_normals,
                               faces_neighbors,
                               faces_num_interior,
                               faces_int_ext)

            cells_num = int(f.get('cells/num')[:].item())
            cells_nodes = f.get('cells/nodes')[:].astype(int) - 1
            cells_centroids = f.get('cells/centroids')[:]
            cells_volumes = f.get('cells/volumes')[:]
            cells_to_hf = np.concatenate((faces_neighbors[:, :faces_num_interior].ravel(),
                                          faces_neighbors[:, faces_num_interior:].max(axis=0)))
            
            self.cells = Cells(cells_num,
                               cells_nodes,
                               cells_centroids,
                               cells_to_hf,
                               cells_volumes)

            nodes_num = int(f.get('nodes/num')[:].item())
            nodes_coords = f.get('nodes/coords')[:]

            self.nodes = Nodes(nodes_num,
                               nodes_coords)

    def areas(self):
        polygons = self.nodes.coords.T[self.cells.nodes.T, :]
        return np.abs(np.sum(polygons[..., 0] * (np.roll(polygons[..., 1], 1, 1) - np.roll(polygons[..., 1], -1, 1)), axis=1)) / 2

    def cellsContain(self, points: npt.NDArray) -> npt.NDArray:
        polygons = self.nodes.coords.T[self.cells.nodes.T, :]
        return np.nonzero(np.all(np.cross(polygons - np.roll(polygons, 1, 1), points[:, None, None, :] - polygons[None, ...]) > 0, 2))[1]

    def anyCellsWithin(self, polygons: npt.NDArray) -> npt.NDArray:
        cells = np.nonzero(np.all(np.cross(polygons - np.roll(polygons, 1, 1),
                                           self.cells.centroids.T[:, None, None, :] - polygons[None, ...]) > 0, 2))
        cands = cells[0][np.argsort(cells[1])].reshape((-1, 4))
        return cands[np.arange(len(cands)), np.random.random_integers(0, 3, cands.shape[0])]

    
def geom_plot(geom: GeomMRST, values: npt.ArrayLike, ax: Axes):

    patches = [mpatches.Polygon(v, closed=True) for v in geom.nodes.coords.T[geom.cells.nodes.T, :]]
    
    p = PatchCollection(patches)
    p.set_array(values)
    ax.add_collection(p)
    
    return p
