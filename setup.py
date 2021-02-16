import io
import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# Package information
NAME = 'GCfit'
VERSION = "0.3.1"

DESCRIPTION = 'Multimass MCMC fitting of Limepy globular cluster analytic model'
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()

# Contributor information
AUTHOR = 'Nolan Dickson'
CONTACT_EMAIL = 'ndickson@ap.smu.ca'

# Installation information
# TODO This is not the right way to get ssptools, currently need to dload first
REQUIRED = [
    "corner"
    "astro-limepy"
    "astropy"
    "emcee"
    "ssptools"
    "schwimmbad"
    "matplotlib"
    "numpy"
    "scipy"
    "h5py"
    "tqdm"
    "gala"
]

REQUIRES_PYTHON = '>=3.7'

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

    packages=['fitter'],
    scripts=['bin/GCfitter', 'bin/view_chain'],

    include_package_data=True,
    package_data={
        "fitter": ["resources/*.hdf5", "resources/*msp.dat"]
    },
)
