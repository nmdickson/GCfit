import io
import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# Package information
NAME = 'GCfit'
VERSION = "0.9"

DESCRIPTION = 'Multimass MCMC fitting of Limepy globular cluster analytic model'
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()

# Contributor information
AUTHOR = 'Nolan Dickson'
CONTACT_EMAIL = 'ndickson@ap.smu.ca'

# Installation information
# TODO document the subrequirements like geos, gsl, etc
REQUIRED = [
    "corner",
    "astro-limepy==0.1.1",  # TODO only until newest version is updated on pip
    "astropy",
    "emcee",
    "ssptools @ git+https://github.com/nmdickson/ssptools.git",
    "schwimmbad",
    "matplotlib",
    "numpy",
    "scipy",
    "h5py",
    "tqdm",
    "dynesty",
    "gala==1.3",
    "shapely",
    "sphinx-toggleprompt"
]

REQUIRES_PYTHON = '>=3.8'

# setup parameters
setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',

    author=AUTHOR,
    author_email=CONTACT_EMAIL,

    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,

    packages=setuptools.find_packages(),
    scripts=[
        'bin/GCfitter', 'bin/view_chain', 'bin/cluster_dump', 'bin/run_summary'
    ],

    include_package_data=True,
    package_data={
        "fitter": ["resources/*.hdf", "resources/*msp.dat"]
    },
)
