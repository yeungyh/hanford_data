# %%
import os
import numpy as np
import numpy.random as npr
import scipy as sp
import h5py
import GPy
from geom_mrst import GeomMRST
from bc_mrst import BCMRST
from darcy import DarcyExp
from tpfa import TPFA

# %%
# Setup
os.chdir(os.path.dirname(os.path.abspath(__file__)))
nyobs = 100
conditional = True
data_path = '../data/'
geom_filename = data_path + f'geom_1x.mat'
bc_filename = data_path + f'bc_1x.mat'
conduct_filename = data_path + f'conduct_log_1x.mat'
well_cells_filename = data_path + f'well_cells_1x.mat'
yobs_filename = data_path + f'yobs_{nyobs}_1x.npy'

# %%
# Load Data Files
with h5py.File(conduct_filename, 'r') as f:
    ytrue = f.get('conduct_log')[:].ravel()

with h5py.File(well_cells_filename, 'r') as f:
    iuobs = f.get('well_cells')[:].ravel() - 1

iyobs = np.load(yobs_filename)
nyobs_set = iyobs.shape[0] if conditional else 1

print(nyobs_set)

# %%
# y Field
ytm = np.mean(ytrue)
yref = ytrue - ytm
yobs = yref[iyobs]

# %%
# u Field
geom = GeomMRST(geom_filename)
bc = BCMRST(geom, bc_filename)
bc.rescale('N', ytm)

Nc = geom.cells.num
Ninf = geom.faces.num_interior

prob = DarcyExp(TPFA(geom, bc), iuobs)
uref = prob.solve(yref)
uobs = uref[iuobs]
Nuobs = iuobs.size

# %%
# Data Processing
std_dev_ref = 1.0
cor_len_ref = 0.1
seed = 0
rs = npr.RandomState(seed)
Nens = 20000
Nxi = 1000

ypred = np.empty((nyobs_set, Nc))
Cypred = np.empty((nyobs_set, Nc, Nc))
Lambda_y = np.empty((nyobs_set, Nxi))
Psi_y = np.empty((nyobs_set, Nc, Nxi))
xi_ens = rs.standard_normal((nyobs_set, Nens, Nxi))
ytms_ens = np.empty((nyobs_set, Nens, Nc))
u_ens = np.empty((nyobs_set, Nens, Nc))

for t in range(nyobs_set):
    iylearn = iyobs[t]
    ylearn = yref[iylearn]

    # Gaussian Process Regression
    mflearn = GPy.mappings.Constant(2, 1, value=1)
    klearn = GPy.kern.sde_Matern52(
        input_dim=2, variance=std_dev_ref**2, lengthscale=cor_len_ref)
    mylearn = GPy.models.GPRegression(
        geom.cells.centroids[:, iylearn].T, ylearn[:, None],
        klearn, noise_var=np.sqrt(np.finfo(float).eps),
        mean_function=mflearn)
    mylearn.optimize(messages=True, ipython_notebook=False)

    if conditional:
        mYref = GPy.models.GPRegression(
            geom.cells.centroids[:, iyobs[t]].T, yobs[t, :, None],
            mylearn.kern, noise_var=np.sqrt(np.finfo(float).eps),
            mean_function=mylearn.constmap)
        ypred[t], Cypred[t] = (lambda x, y: (x.ravel(), y))(
            *mYref.predict_noiseless(geom.cells.centroids.T, full_cov=True))
    else:
        ypred[t] = mylearn.constmap.C.values[0] * np.ones_like(yref)
        Cypred[t] = mylearn.kern.K(geom.cells.centroids.T)

    # Eigendecomposition
    Lambda_y[t], Psi_y[t] = (lambda l, p: (l[::-1], (p.real @ np.diag(np.sqrt(np.abs(l))))[:, ::-1]))(
        *sp.linalg.eigh(Cypred[t], eigvals=(Nc - Nxi, Nc - 1)))

    # Solving for u from KLE of y
    for i in range(Nens):
        ytms_ens[t, i] = ypred[t] + Psi_y[t] @ xi_ens[t, i]
        u_ens[t, i] = prob.solve(ytms_ens[t, i])

# %%
# Save to File
with h5py.File(data_path + f"hanford_ens_data_{'cond' if conditional else 'uncond'}.h5", 'w') as f:
    f.create_dataset('xi_ens', data=xi_ens)
    f.create_dataset('ytm', data=ytm)
    f.create_dataset('yref', data=yref)
    f.create_dataset('Nens', data=Nens)
    f.create_dataset('Nxi', data=Nxi)

    if conditional:
        f.create_dataset('nyobs', data=nyobs)
        f.create_dataset('nyobs_set', data=nyobs_set)

    for t in range(nyobs_set):
        g = f.create_group(f't{t}') if conditional else f
        g.create_dataset('iyobs', data=iyobs[t])
        g.create_dataset('ypred', data=ypred[t])
        g.create_dataset('Psi_y', data=Psi_y[t])
        g.create_dataset('ytms_ens', data=ytms_ens[t])
        g.create_dataset('u_ens', data=u_ens[t])

# %%
