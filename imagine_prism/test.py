# -*- coding: utf-8 -*-


# %% IMPORTS
# Package imports
import astropy.units as apu
import numpy as np


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
    # Package imports
    from imagine.fields import (
        CosThermalElectronDensityFactory, NaiveGaussianMagneticFieldFactory,
        UniformGrid)
    from imagine.likelihoods import EnsembleLikelihood
    from imagine.observables import Covariances, Measurements, TabularDataset
    from imagine.pipelines import UltranestPipeline
    from imagine.priors import FlatPrior
    from imagine.simulators import TestSimulator

    # IMAGINE-PRISM imports
    from imagine_prism import PRISMPipeline

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
    pipe = PRISMPipeline(img_pipe, root_dir='tests', working_dir='imagine')
