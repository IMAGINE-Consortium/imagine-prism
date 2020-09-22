# -*- coding: utf-8 -*-


# %% IMPORTS
# Built-in imports
import os
from os import path

# Package imports
import astropy.units as apu
import e13tools as e13
from imagine.pipelines import Pipeline as imagine_Pipeline
import numpy as np
from prism import Pipeline as prism_Pipeline

# IMAGINE-PRISM imports
from imagine_prism.modellink import IMAGINELink


# %% FUNCTION DEFINITIONS
# Function factory that returns special PRISMPipeline class instances
def get_PRISMPipeline_obj(imagine_pipeline_obj, *args, **kwargs):
    # Save provided imagine_pipeline_obj
    img_pipe = imagine_pipeline_obj

    # Make tuple of overridden attributes
    overridden_attrs = ('__init__', 'call')

    # %% PRISMPIPELINE CLASS DEFINITION
    class PRISMPipeline(imagine_Pipeline):
        # Override constructor
        def __init__(self, *args, **kwargs):

            # Initialize IMAGINELink
            modellink_obj = IMAGINELink(img_pipe)

            # Initialize PRISM Pipeline
            self._prism_pipe = prism_Pipeline(modellink_obj, *args, **kwargs)

            # Store PRISM's communicator
            self._prism_comm = self._prism_pipe._comm

            # Create a directory for storing chains for IMAGINE and set it
            chain_dir = path.join(self._prism_pipe._working_dir,
                                  'imagine_chains')

            # Controller creates directory if necessary
            if self._prism_pipe._is_controller and not path.exists(chain_dir):
                os.mkdir(chain_dir)
            self._prism_comm.Barrier()

            # Set directory
            img_pipe.chains_directory = chain_dir

        # If requested attribute is not a method, use img_pipe for getattr
        def __getattribute__(self, name):
            if name not in overridden_attrs and name in img_pipe.__dir__():
                return(getattr(img_pipe, name))
            else:
                return(super().__getattribute__(name))

        # If requested attribute is not a method, use img_pipe for setattr
        def __setattr__(self, name, value):
            if name not in overridden_attrs and name in img_pipe.__dir__():
                setattr(img_pipe, name, value)
            else:
                super().__setattr__(name, value)

        # If requested attribute is not a method, use img_pipe for delattr
        def __delattr__(self, name):
            if name not in overridden_attrs and name in img_pipe.__dir__():
                delattr(img_pipe, name)
            else:
                super().__delattr__(name)

        # %% CLASS PROPERTIES
        @property
        def prism_pipe(self):
            """
            :obj:`prism.Pipeline`: The PRISM Pipeline object that is used in
            this :obj:`~PRISMPipeline` object.

            """

            return(self._prism_pipe)

        # %% USER METHODS
        # Override call
        def call(self, *args, **kwargs):
            self._img_pipe.call(*args, **kwargs)

    # %% REMAINDER OF FUNCTION FACTORY
    # Initialize PRISMPipeline
    pipe = PRISMPipeline(*args, **kwargs)

    # Return it
    return(pipe)


# %% FUNCTION DEFINITIONS
# This function returns the mock data
def get_mock_data():
    a0 = 3. # true value of a in microgauss
    b0 = 6. # true value of b in microgauss
    e = 0.1 # std of gaussian measurement error
    s = 233 # seed fixed for signal field

    size = 10 # data size in measurements
    x = np.linspace(0.01,2.*np.pi-0.01,size) # where the observer is looking at

    np.random.seed(s) # set seed for signal field

    signal = (1+np.cos(x)) * np.random.normal(loc=a0,scale=b0,size=size)

    fd = signal + np.random.normal(loc=0.,scale=e,size=size)

    # We load these to an astropy table for illustration/visualisation
    data = {'meas' : apu.Quantity(fd, apu.microgauss*apu.cm**-3),
            'err': np.ones_like(fd)*e,
            'x': x,
            'y': np.zeros_like(fd),
            'z': np.zeros_like(fd)}

    return(data)


# %% MAIN SCRIPT
if __name__ == '__main__':
    from imagine.fields import (
        CosThermalElectronDensityFactory, NaiveGaussianMagneticFieldFactory,
        UniformGrid)
    from imagine.likelihoods import EnsembleLikelihood
    from imagine.observables import Covariances, Measurements, TabularDataset
    from imagine.pipelines import UltranestPipeline
    from imagine.priors import FlatPrior
    from imagine.simulators import TestSimulator

    # Obtain data
    data = get_mock_data()

    # Create dataset
    mock_dataset = TabularDataset(data, name='test', data_col='meas',
                                  err_col='err')

    # Create measurements and covariances objects
    mock_data = Measurements(mock_dataset)
    mock_cov = Covariances(mock_dataset)

    # Create grid
    grid = UniformGrid(box=[[0, 2*np.pi]*apu.kpc,
                            [0, 0]*apu.kpc,
                            [0, 0]*apu.kpc],
                       resolution=[30, 1, 1])

    # Create factory for electron density field
    ne_factory = CosThermalElectronDensityFactory(grid=grid)
    ne_factory.default_parameters = {'a': 1*apu.rad/apu.kpc,
                                     'beta': np.pi/2*apu.rad,
                                     'gamma': np.pi/2*apu.rad}

    # Create factory for magnetic field
    B_factory = NaiveGaussianMagneticFieldFactory(grid=grid)
    B_factory.active_parameters = ('a0', 'b0')
    B_factory.priors = {'a0': FlatPrior(interval=[-5, 5]*apu.microgauss),
                        'b0': FlatPrior(interval=[0, 10]*apu.microgauss)}

    # Combine factories together
    factories = [ne_factory, B_factory]

    # Create simulator
    simulator = TestSimulator(mock_data)

    # Create likelihood
    likelihood = EnsembleLikelihood(mock_data, mock_cov)

    # Create pipeline
    img_pipe = UltranestPipeline(simulator=simulator,
                                 factory_list=factories,
                                 likelihood=likelihood,
                                 ensemble_size=150)

    # Create PRISMPipeline object
    pipe = get_PRISMPipeline_obj(img_pipe, root_dir='tests',
                                 working_dir='imagine')
