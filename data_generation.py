import numpy as np
import scipy as sp
import h5py
import GPy
import argparse
from numpy.random import RandomState
from sdfs.geom_mrst import GeomMRST
from sdfs.bc_mrst import BCMRST
from sdfs.darcy import DarcyExp
from sdfs.tpfa import TPFA
from tqdm import trange

parser = argparse.ArgumentParser(description='Dataset generation')
parser.add_argument('--cond', action='store_true')
parser.add_argument('--nyobs', type=int, choices=[10, 25, 50, 100, 200], default=100)
args = parser.parse_args()
conditional = args.cond
nyobs = args.nyobs
print(f"Generating {'conditional' if conditional else 'unconditional'} data for Nyobs = {nyobs:d}...")

data_path = 'data/'
geom_filename = data_path + f'geom_1x.mat'
bc_filename = data_path + f'bc_1x.mat'
conduct_filename = data_path + f'conduct_log_1x.mat'
well_cells_filename = data_path + f'well_cells_1x.mat'
yobs_filename = data_path + f'yobs_{nyobs}_1x.npy'

# Load Data Files
with h5py.File(conduct_filename, 'r') as f:
    ytrue = f.get('conduct_log')[:].ravel()

with h5py.File(well_cells_filename, 'r') as f:
    iuobs = f.get('well_cells')[:].ravel() - 1

iyobs = np.load(yobs_filename)
nyobs_set = iyobs.shape[0] if conditional else 1

# Log-transmissivity field y 
ytm = np.mean(ytrue)
yref = ytrue - ytm
yobs = yref[iyobs]

# Hydraulic pressure field u
geom = GeomMRST(geom_filename)
bc   = BCMRST(geom, bc_filename)
bc.rescale('N', ytm)

Nc    = geom.cells.num
Ninf  = geom.faces.num_interior
prob  = DarcyExp(TPFA(geom, bc), iuobs)
uref  = prob.solve(yref)
uobs  = uref[iuobs]
Nuobs = iuobs.size

# Data Processing
std_dev_ref = 1.0
cor_len_ref = 0.1
seed = 0
rs = RandomState(seed)
Nens = 20000
Nxi = 1000

xi_ens   = rs.standard_normal((Nens, Nxi))
ytms_ens = np.empty((Nens, Nc))
u_ens    = np.empty((Nens, Nc))

out_file = data_path + f"hanford_data_{nyobs:d}_{'cond' if conditional else 'uncond'}.h5"

with h5py.File(out_file, 'w') as f:
    f.create_dataset('xi_ens', data=xi_ens)
    f.create_dataset('ytm', data=ytm)
    f.create_dataset('yref', data=yref)
    f.create_dataset('uref', data=uref)
    f.create_dataset('Nens', data=Nens)
    f.create_dataset('Nxi', data=Nxi)
    f.create_dataset('nyobs', data=nyobs)
    f.create_dataset('nyobs_set', data=nyobs_set)

for t in range(nyobs_set):
    print(f"Dataset {t + 1:d} of {nyobs_set:d}:")
    iylearn = iyobs[t]
    ylearn = yref[iylearn]

    # Gaussian Process Regression
    print("\nGP regression...")
    mflearn = GPy.mappings.Constant(2, 1, value=1)
    klearn = GPy.kern.sde_Matern52(input_dim=2, variance=std_dev_ref**2, lengthscale=cor_len_ref)
    mylearn = GPy.models.GPRegression(geom.cells.centroids[:, iylearn].T, ylearn[:, None], klearn,
                                      noise_var=np.sqrt(np.finfo(float).eps), mean_function=mflearn)
    mylearn.optimize(messages=True, ipython_notebook=False)
    print("Calibrated GP parameters:")
    print(klearn)
    print(mflearn)

    if conditional:
        mYref = GPy.models.GPRegression(geom.cells.centroids[:, iyobs[t]].T, yobs[t, :, None],
                                        mylearn.kern, noise_var=np.sqrt(np.finfo(float).eps),
                                        mean_function=mylearn.constmap)
        ypred, Cypred = (lambda x, y: (x.ravel(), y))(
            *mYref.predict_noiseless(geom.cells.centroids.T, full_cov=True))
    else:
        ypred = mylearn.constmap.C.values[0] * np.ones_like(yref)
        Cypred = mylearn.kern.K(geom.cells.centroids.T)

    # Eigendecomposition
    Lambda_y, Psi_y = (lambda l, p: (l[::-1], (p.real @ np.diag(np.sqrt(np.abs(l))))[:, ::-1]))(
        *sp.linalg.eigh(Cypred, eigvals=(Nc - Nxi, Nc - 1)))

    # Solving for u from KLE of y
    print("\nCalculating dataset...")
    for i in trange(Nens):
        ytms_ens[i] = ypred + Psi_y @ xi_ens[i]
        u_ens[i] = prob.solve(ytms_ens[i])

    with h5py.File(out_file, 'a') as f:
        g = f.create_group(f't{t}') if conditional else f
        g.create_dataset('iyobs', data=iyobs)
        g.create_dataset('ypred', data=ypred)
        g.create_dataset('Psi_y', data=Psi_y)
        g.create_dataset('ytms_ens', data=ytms_ens)
        g.create_dataset('u_ens', data=u_ens)
