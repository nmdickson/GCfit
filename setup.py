import io
import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# Package information
NAME = 'GCfit'
VERSION = "0.10"

DESCRIPTION = 'Fitting of multimass equilibrium models of globular clusters'
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()

# Contributor information
AUTHOR = 'Nolan Dickson'
CONTACT_EMAIL = 'nolan.dickson@smu.ca'

# Installation information
# TODO document the subrequirements like geos, gsl, etc
REQUIRED = [
    "corner",
    # TODO only until newest version is updated on pip
    "astro-limepy @ git+https://github.com/mgieles/limepy.git",
    "astropy",
    "emcee",
    "ssptools @ git+https://github.com/SMU-clusters/ssptools.git",
    "schwimmbad",
    "matplotlib",
    "numpy",
    "scipy",
    "h5py",
    "tqdm",
    "dynesty",
    "gala",
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
        'bin/GCfitter', 'bin/generate_model_CI'
    ],

    include_package_data=True,
    package_data={
        "fitter": ["resources/*.hdf", "resources/*msp.dat"]
    },
)
