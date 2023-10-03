# Generation synthetic Hanford Site data

This repository contains the code employed for generating the synthetic Hanford Site datasets hosted
in LOCATION. These datasets contain 20,000 realizations of the log-hydraulic transmissivity field of
a two-dimensional stationary, saturated flow model of the Hanford Site, together with the
corresponding hydraulic pressure fields.

The log-hydraulic transmissivity fields are constructed by computing and sampling the
1,000-dimensional truncated Kosambi-Karhunen-Loève (KKL) expansion of a stationary Gaussian Process
Regression (GPR)/Kriging model calibrated using `nyobs` spatially sparse observations of the
reference log-hydraulic conductivity field.

The data repository contains the datasets listed below. For "conditional" datasets we condition the
GPR model using the `nyobs` log-transmissivity observations before computing the KKL. Therefore, for
unconditional datasets, the variance of the log-transmissivity realizations is constant and equal to
the calibrated GPR variance, whereas for conditional models the variance at the observation
locations is close to zero (but not equal to zero due to truncation). The maximum variance at
observation locations for the conditional datasets is less than 2e-13.

| Dataset                      | Conditional? | `nyobs` |
|------------------------------|--------------|---------|
| `hanford_data_100_uncond.h5` | No           | 100     |
| `hanford_data_200_uncond.h5` | No           | 200     |
| `hanford_data_100_cond.h5`   | Yes          | 100     |

These datasets were generated using the following commands:
```
python data_generation.py
python data_generation.py --cond
python data_generation.py --nyobs 200
```

## Dataset structure

The datasets are generated in HD5F format. All datasets have the following fields in the root:

| Field       | Description                                                      |
|-------------|------------------------------------------------------------------|
| `Nens`      | Number of realizations (= 20,000)                                |
| `Nxi`       | Number of Kosambi-Karhunen-Loève (KKL) expansion terms (= 1,000) |
| `xi_ens`    | `Nens` vectors of KKL coefficients                               |
| `yref`      | Reference log-transmissivity field minus its mean                |
| `ytm`       | Reference log-transmissivity field's mean                        |
| `uref`      | Reference hydraulic conductivity field                           |
| `nyobs`     | Number of transmissivity observation locations                   |
| `nyobs_set` | Number of sets of transmissivity observation locations.          |

`xi_ens` is the same for all the datasets in the data repository.

For unconditional datasets, the following fields are in the root, whereas for conditional datasets
they are in groups named `t[0-nyobs_set]` (e.g., `t0`, `t1`, etc.):

| Field            | Description                                                                                                                    |
|------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `iyobs`          | `nyobs` indices corresponding to the y observation locations                                                                   |
| `u_ens`          | `Nens` vectors of discretized pressure fields                                                                                  |
| `ytms_ens`       | `Nens` vectors of discretized log-transmissivity fields, corresponding to the entries of `xi_ens`, minus the true field's mean |
| `ypred`, `Psi_y` | The KKL mean and matrix of coefficients                                                                                        |

The entries of `xi_ens` and `ytms_ens` are related by the KKL as

```
ytms_ens[i] = ypred + Psi_y @ xi_ens[i]
```

For any realization and for the reference, the log-transmissivity fields can be recovered by adding
the true field's mean:

```python
ytm + yref        # The true log-transmissivity field
ytm + ytms_ens[i] # The ith log-transmissivity field in the dataset
```

Please see the notebook file
[`notebooks/data_read_example.ipynb`](notebooks/data_read_example.ipynb) for examples of how to read
the data.

## Creating other datasets

The script `data_generation.py` can be used to create other datasets (conditional or unconditiona)
with different values of `nyobs` (10, 25, 50, 100, and 200). Datasets created from scratch may
differ from the ones hosted in the data repository because of different machines generate random
numbers using the same code. To create fully reproducible datasets beyond the ones provided we
recommend you read `xi_ens` from one of the provided datasets and use it to create new data.
