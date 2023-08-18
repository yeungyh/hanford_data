# Generation and Processing of Hanford Site Data

## Instructions

1. Clone this repository and navigate to your local copy

2. Install [Miniforge](https://github.com/conda-forge/miniforge) and
   choose to *not* initialize Miniforge by running `conda init`. No
   need to polute your runcom files. For macOS and bash this looks
   like:
   
   a. Download the installation script and run it:
      ```
	  bash Miniforge3-MacOSX-arm64.sh
	  ```
	  
   b. Accept the license
   
   c. Specify an installation location
   
   d. Say *no* to `Do you wish the installer to initialize Miniforge3 by running conda init?`
   
	
3. Initialize `conda`. For the previous example, this looks like
   ```
   eval "$(MINIFORGE3_INSTALL_DIR/bin/conda shell.bash hook)"
   ```
	
4. Create the `conda-forge` environment `hanford_data`:

   ```
   conda env create -f environment.yml
   ```
	
5. Activate the `hanford_data` environment

   ```
   conda activate hanford_data
   ```
	
6. Download the data file `mpi23_ens_data.h5` from the [Zenodo repository](https://zenodo.org/record/8027857) and place it on the `data` folder

7. Verify the checksums
   ```
   cd data; sha256sum -c SHA256SUMS; cd ..
   ```

8. Run `data_generation.py`
   ```
   python3 data_generation
   ```
	
Checkout the notebook file [`notebooks/data_read_example.ipynb`](notebooks/data_read_example.ipynb) to see how to read the data and start working with it.

## HDF5 data file

This file contains the following datasets:
- `Nens`: Number of data pairs
- `Nxi`: Number of terms of the Kosambi-Karhunen-Loève (KKL) expansion of the log-transmissivity field
- `xi_ens`: `Nens` vectors of KKL coefficients
- `yref`: The true log-transmissivity field minus its mean
- `nyobs`: Number of y observation locations
- `nyobs_set`: Number of sets of y observation locations.
For conditional dataset, each set of y observation locations has a group `t[0-9]`. Each group has the following datasets:
- `iyobs`: `nyobs` indices corresponding to the y observation locations.
- `u_ens`: `Nens` vectors of discretized pressure fields
- `ytms_ens`: `Nens` vectors of discretized log-transmissivity fields, corresponding to the entries of `xi_ens`, minus the true field's mean
- `ytm`: The true log-transmissivity field's mean
- `ypred`, `Psi_y`: The KKL mean and matrix of coefficients
For unconditional dataset, there is no group and all datasets are under the root. The above dataset except `iyobs` are under the root.

The entries of `xi_ens` and `ytms_ens` are related by the KKL:
```
ytms_ens[i] = ypred + Psi_y @ xi_ens[i]
```

Any of the log-transmissivity fields can be recovered by adding the true field's mean:
```python
ytm + yref        # The true log-transmissivity field
ytm + ytms_ens[i] # The ith log-transmissivity field in the dataset
```

