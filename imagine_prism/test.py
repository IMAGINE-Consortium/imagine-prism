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
def tutorial_one():
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
    B_factory.priors = {'a0': FlatPrior(*[-5, 5]*apu.microgauss),
                        'b0': FlatPrior(*[0, 10]*apu.microgauss)}

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
    pipe = PRISMPipeline(img_pipe, root_dir='tests', working_dir='imagine_1')

    # Return pipe
    return(pipe)


def tutorial_five():
    # Package imports
    import os
    # External packages
    import numpy as np
    import healpy as hp
    import astropy.units as u
    import corner
    import matplotlib.pyplot as plt
    import cmasher as cmr
    # IMAGINE
    import imagine as img
    import imagine.observables as img_obs
    ## WMAP field factories
    from imagine.fields.hamx import BregLSA, BregLSAFactory
    from imagine.fields.hamx import TEregYMW16, TEregYMW16Factory
    from imagine.fields.hamx import CREAna, CREAnaFactory

    # IMAGINE-PRISM imports
    from imagine_prism import PRISMPipeline

    ## Sets the resolution
    nside=2
    size = 12*nside**2

    # Generates the fake datasets
    sync_dset = img_obs.SynchrotronHEALPixDataset(data=np.empty(size)*u.K,
                                                  frequency=23, typ='I')
    fd_dset = img_obs.FaradayDepthHEALPixDataset(data=np.empty(size)*u.rad/u.m**2)

    # Appends them to an Observables Dictionary
    trigger = img_obs.Measurements(sync_dset, fd_dset)

    # Prepares the Hammurabi simmulator for the mock generation
    mock_generator = img.simulators.Hammurabi(measurements=trigger)

    # BregLSA field
    breg_lsa = BregLSA(parameters={'b0':3, 'psi0': 27.0, 'psi1': 0.9, 'chi0': 25.0})

    # CREAna field
    cre_ana = CREAna(parameters={'alpha': 3.0, 'beta': 0.0, 'theta': 0.0,
                                 'r0': 5.0, 'z0': 1.0,
                                 'E0': 20.6, 'j0': 0.0217})

    # TEregYMW16 field
    tereg_ymw16 = TEregYMW16(parameters={})

    ## Generate mock data (run hammurabi)
    outputs = mock_generator([breg_lsa, cre_ana, tereg_ymw16])

    ## Collect the outputs
    mockedI = outputs[('sync', 23.0, nside, 'I')].global_data[0]
    mockedRM = outputs[('fd', None, nside, None)].global_data[0]
    dm=np.mean(mockedI)
    dv=np.std(mockedI)

    ## Add some noise that's just proportional to the average sync I by the factor err
    err=0.01
    dataI = (mockedI + np.random.normal(loc=0, scale=err*dm, size=size)) << u.K
    errorI = ((err*dm)**2) << u.K
    sync_dset = img_obs.SynchrotronHEALPixDataset(data=dataI, error=errorI,
                                                  frequency=23, typ='I')

    ## Just 0.01*50 rad/m^2 of error for noise.
    dataRM = (mockedRM + np.random.normal(loc=0.,scale=err*50.,size=12*nside**2))*u.rad/u.m/u.m
    errorRM = ((err*50.)**2) << u.rad/u.m**2
    fd_dset = img_obs.FaradayDepthHEALPixDataset(data=dataRM, error=errorRM)

    mock_data = img_obs.Measurements(sync_dset, fd_dset)

    ## Use an ensemble to estimate the galactic variance
    likelihood = img.likelihoods.EnsembleLikelihood(mock_data)

    breg_factory = BregLSAFactory()
    breg_factory.active_parameters = ('b0', 'psi0')
    breg_factory.priors = {'b0':  img.priors.FlatPrior(0, 10),
                          'psi0': img.priors.FlatPrior(0, 50)}
    breg_factory.default_parameters = {'b0': 3,
                                       'psi0': 27}
    ## Fixed CR model
    cre_factory = CREAnaFactory()
    ## Fixed FE model
    fereg_factory = TEregYMW16Factory()

    # Final Field factory list
    factory_list = [breg_factory, cre_factory, fereg_factory]

    simulator = img.simulators.Hammurabi(measurements=mock_data)

    # Assembles the pipeline using MultiNest as sampler
    pipeline = img.pipelines.MultinestPipeline(simulator=simulator,
                                               factory_list=factory_list,
                                               likelihood=likelihood,
                                               ensemble_size=1)
    pipeline.sampling_controllers = {'n_live_points': 50}

    # Create PRISMPipeline object
    pipe = PRISMPipeline(pipeline, root_dir='/home/x1313e/stack/PhD/PRISM_Root/tests', working_dir='imagine_5',
                         prism_par={'n_sam_init': 100})

    # Return pipe
    return(pipe)
