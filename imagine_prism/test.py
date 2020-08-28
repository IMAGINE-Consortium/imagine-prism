# -*- coding: utf-8 -*-


# %% IMPORTS
# Package imports
import astropy.units as apu
import e13tools as e13
from imagine.fields import (
    CosThermalElectronDensityFactory, NaiveGaussianMagneticFieldFactory,
    UniformGrid)
from imagine.likelihoods import Likelihood, EnsembleLikelihood
from imagine.observables import Covariances, Measurements, TabularDataset
from imagine.pipelines import UltranestPipeline
from imagine.priors import FlatPrior
from imagine.simulators import TestSimulator
import numpy as np
from prism import Pipeline
from prism.modellink import ModelLink


# %% CLASS DEFINITIONS
# Define the IMAGINELink class
class IMAGINELink(ModelLink):
    # Override constructor
    def __init__(self, imagine_pipeline_obj):
        # Save provided imagine_pipeline_obj
        self._img_pipe_obj = imagine_pipeline_obj

        # Obtain model_parameters and model_data
        model_parameters = self._get_model_parameters()
        model_data = self._get_model_data()

        # Call super constructor
        super().__init__(model_parameters=model_parameters,
                         model_data=model_data)

    # This function retrieves the model parameters from the IMAGINE Pipeline
    def _get_model_parameters(self):
        # Create empty dict of model parameters
        model_par = {}

        # Loop over all factories in the Pipeline
        for factory in self._img_pipe_obj._factory_list:
            # Loop over all active parameters in this factory
            for par in factory.active_parameters:
                # Obtain the range and estimate for this parameter
                rng = factory.parameter_ranges[par].to_value()
                est = factory.default_parameters[par].to_value()

                # Add this parameter to the dict
                model_par[par] = [*rng, est]

        # Return model_par
        return(model_par)

    # This function retrieves the model data from the IMAGINE Pipeline
    def _get_model_data(self):
        # Create empty dict of model data
        model_data = {}

        # Obtain the measurements and covariances dicts
        meas_dct = self._img_pipe_obj._likelihood._measurement_dict
        cov_dct = self._img_pipe_obj._likelihood._covariance_dict

        # Obtain the keys of all datasets
        keys = list(meas_dct._archive.keys())

        # Loop over all keys
        for key in keys:
            # Obtain the corresponding measurements and covariances
            meas = meas_dct[key]
            cov = cov_dct[key]

            # Obtain the data values
            data_val = meas.data[0]

            # Obtain the data errors
            data_err = np.diag(cov.data)

            # Convert coordinates into a format for PRISM
            coords_type, coords_list = self.convert_coords(meas.coords)

            # Add every data point individually
            for coord, val, err in zip(coords_list, data_val, data_err):
                # Construct full data_idx
                idx = (*key, coords_type, *coord)

                # Add data point to model_data
                model_data[idx] = [val, err]

        # Return model_data
        return(model_data)

    # This function converts the coords dict to a proper format
    def convert_coords(self, coords):
        # Determine the coordinates type
        coords_type = coords['type']

        # Extract the coordinates accordingly
        if(coords_type == 'cartesian'):
            coords_list = list(zip(coords['x'].to_value(),
                                   coords['y'].to_value(),
                                   coords['z'].to_value()))
        else:
            coords_list = list(zip(coords['lon'].to_value(),
                                   coords['lat'].to_value()))

        # Return coords_type and coords_list
        return(coords_type, coords_list)

    # Override call_model
    def call_model(self, emul_i, par_set, data_idx):
        # Obtain the names of all parameters
        par_names = self._img_pipe_obj.get_par_names()

        # Obtain the sorting order
        index = list(map(self._par_name.index, par_names))

        # Convert the provided par_dict to a par_set
        par_set = np.array(par_set.values())[index]

        # Convert par_set to unit_space
        par_set = self._to_unit_space(par_set)

        # Evaluate the IMAGINE pipeline
        sims = self._img_pipe_obj._get_observables(par_set)

        # Create empty dict of model_data
        mod_dict = {}

        # Loop over all observables
        for key, obs in sims.archive.items():
            # Convert coordinates into a format for PRISM
            coords_type, coords_list = self.convert_coords(obs.coords)

            # Loop over all values with appropriate coordinates
            # TODO: Currently only uses the first ensemble
            for coord, val in zip(coords_list, obs.data[0]):
                # Obtain data_idx
                idx = (*key, coords_type, *coord)

                # Add value
                mod_dict[idx] = val

        # Return mod_dict
        return(mod_dict)

    # Override get_md_var
    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


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
    pipeline = UltranestPipeline(simulator=simulator,
                                 factory_list=factories,
                                 likelihood=likelihood,
                                 ensemble_size=150)

    # Create IMAGINELink object
    modellink_obj = IMAGINELink(pipeline)
