import numpy as np
from numpy.typing import NDArray
from scipy.sparse import bsr_array
from sdfs.geom_mrst import GeomMRST
from typing import NamedTuple

class Map2GridMap(NamedTuple):
    cells: NDArray
    coords: NDArray
    sets: list

class AmbiguityException(Exception):
    """Raise when it's not clear what's the cardinal direction of a neighbor relation"""

class MaxIterError(Exception):
    """Raise when number of iterations is exceeded"""

def compute_map(geom: GeomMRST, max_iter: int=100) -> Map2GridMap:
    """
    Computes a map from a geometry of quadrilateral cells to a regular grid.

    Parameters
    ----------
    geom : GeomMRST
        MRST-compatible geometry of quadrilaterals
    max_iter: int
        Maximum number of iterations

    Returns
    -------
    map : Map2GridMap
        Object containing all the mapping information to be used with 'map2grid'
    """
    
    touched  = np.full((geom.cells.num,), False, dtype=bool)
    conflict = np.full((geom.cells.num,), False, dtype=bool)
    px  = np.full((geom.cells.num,), np.nan)
    py  = np.full((geom.cells.num,), np.nan)
    pxc = np.full((geom.cells.num,), np.nan)
    pyc = np.full((geom.cells.num,), np.nan)

    px[0] = 0
    py[0] = 0
    touched[0] = True

    iter = 0
    while np.any(touched == False):
        for cell in range(geom.cells.num):
            neighbors = np.unique(geom.faces.neighbors[:, np.argwhere(np.logical_or(geom.faces.neighbors[0] == cell, geom.faces.neighbors[1] == cell)).flatten()].flatten())
            neighbors = neighbors[np.logical_and(neighbors != -1, neighbors != cell)]
            if touched[cell] == False:
                continue
            for neighbor in neighbors:
                npx = px[cell]; npy = py[cell]
                dx = geom.cells.centroids[0, neighbor] - geom.cells.centroids[0, cell]
                dy = geom.cells.centroids[1, neighbor] - geom.cells.centroids[1, cell]
                if dy > 0 and np.abs(dy) > np.abs(dx):
                    npx = px[cell]; npy = py[cell] + 1
                elif dy < 0 and np.abs(dy) > np.abs(dx):
                    npx = px[cell]; npy = py[cell] - 1
                elif dx > 0 and np.abs(dx) > np.abs(dy):
                    npx = px[cell] + 1; npy = py[cell]
                elif dx < 0 and np.abs(dx) > np.abs(dy):
                    npx = px[cell] - 1; npy = py[cell]
                else:
                    raise AmbiguityException(f'Ambiguous direction from cells {cell} to {neighbor}. dx: {dx} dy: {dy}')
                if touched[neighbor]:
                    if npx != px[neighbor] or npy != py[neighbor]:
                        conflict[neighbor] = True
                        pxc[neighbor] = npx; pyc[neighbor] = npy
                else:
                    px[neighbor] = npx; py[neighbor] = npy
                    touched[neighbor] = True
        iter = iter + 1
        if iter > max_iter:
            raise MaxIterError

    cells = np.concatenate((np.arange(geom.cells.num), np.argwhere(conflict==True).flatten()))
    pxfull = np.concatenate((px, pxc[conflict]))
    pyfull = np.concatenate((py, pyc[conflict]))
    coords = np.vstack((pxfull - pxfull.min(), pyfull - pyfull.min())).T.astype(int)
    unique = np.unique(coords, axis=0)
    sets = [(coords == i).all(axis=1).nonzero()[0] for i in unique]

    return Map2GridMap(cells=cells, coords=unique, sets=sets)

def map2grid(data: NDArray, map: Map2GridMap) -> NDArray:
    """
    Map 1-D array to a regular grid.

    Parameters
    ----------
    data : NDArray
        Vector of data to map
    map : Map2GridMap
        Object containing mapping information, computed using 'compute_map'

    Returns
    -------
    data_mapped: NDArray
        Matrix of mapped data
    """
    data_avg = np.array([np.mean(data[map.cells[set]]) for set in map.sets])
    return bsr_array((data_avg, map.coords.T)).T.todense()




                    
