# GCfit

## Fitting of static equilibrium globular cluster models

Python package enabling the generalized fitting of globular cluster
observations to distribution function based lowered isothermal
([LIMEPY](https://github.com/mgieles/limepy)) models to various observational
data products, via a parallelized [MCMC](https://github.com/dfm/emcee/) suite.

## Installation

The `fitter` package can be easily installed from this repository using pip.
A fork of the `ssptools` library must be installed seperately, as below. All
other requisite packages will be installed automatically.

```
pip install git+https://github.com/nmdickson/ssptools.git
pip install git+https://github.com/nmdickson/GCfit.git
```

or this repo can be cloned locally and installed with pip.

## Usage

GCfit has two main functionalities; to fit globular cluster models, and to
explore the results of those fitting runs.

### Fitting

Fitting of the clusters is done through the `fit` function of `fitter`.
For ease of use, a command-line script is provided in `GCfitter`, which should
be automatically placed in your path upon installation.

`GCfitter` takes the name of the cluster you wish to fit, as well as a
number of sampler directives. See `GCfitter -h` for a full list of arguments.

#### Examples

Fitting cluster "NGC0104" (also known as 47 Tucanae) for 2000 iterations of 100
MCMC walkers, parallelized locally over 2 CPUs (default). Sampler output is
saved to `~/.GCfit/47Tuc_sampler.hdf` (default).
```
GCfitter 47Tuc -N 2000 --Nwalkers 100 --verbose
```

Fitting cluster "NGC6397" for 1850 iterations of 150 walkers (default), using
MPI. MPI parameters and allocations must be handled by your job
script/scheduler separately. Output and debug info is saved to a local `results`
folder.
```
srun GCfitter NGC6397 -N 1850 --mpi --savedir ./results --debug
```

Fitting cluster "NGC0104" for 2000 iterations of 150 walkers, with custom prior
bounds specified in the file `alt_bounds`, parallelized locally with 4 CPUs.
```
echo '{"a3": [">=", "a2"], "delta": [0.45, 0.5]}' > alt_bounds.json
GCfitter NGC104 -N 2000 --Ncpu 4 --bounds alt_bounds.json --debug
```

### Investigating Fitting Results

TODO