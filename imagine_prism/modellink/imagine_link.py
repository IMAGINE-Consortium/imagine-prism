# -*- coding: utf-8 -*-


# %% IMPORTS
# Package imports
import numpy as np
from prism.modellink import ModelLink

# All declaration
__all__ = ['IMAGINELink']


# %% CLASS DEFINITIONS
# Define the IMAGINELink class
class IMAGINELink(ModelLink):
    # Override constructor
    def __init__(self, imagine_pipeline_obj):
        # Save provided imagine_pipeline_obj
        self._img_pipe = imagine_pipeline_obj

        # Obtain model_parameters and model_data
        model_parameters = self._get_model_parameters()
        model_data = self._get_model_data()

        # Call super constructor
        super().__init__(model_parameters=model_parameters,
                         model_data=model_data)

        # Obtain the parameter names in IMAGINE
        par_names = self._img_pipe.get_par_names()

        # Obtain the sorting order PRISM->IMAGINE
        self._par_index = list(map(self._par_name.index, par_names))

    # This function retrieves the model parameters from the IMAGINE Pipeline
    def _get_model_parameters(self):
        # Create empty dict of model parameters
        model_par = {}

        # Loop over all factories in the Pipeline
        for factory in self._img_pipe._factory_list:
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
        meas_dct = self._img_pipe._likelihood._measurement_dict
        cov_dct = self._img_pipe._likelihood._covariance_dict

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
            coords_type, coords_list = self._convert_coords(meas.coords)

            # Add every data point individually
            for coord, val, err in zip(coords_list, data_val, data_err):
                # Construct full data_idx
                idx = (*key, coords_type, *coord)

                # Add data point to model_data
                model_data[idx] = [val, err]

        # Return model_data
        return(model_data)

    # This function converts the coords dict to a proper format
    def _convert_coords(self, coords):
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
        # Convert the provided par_dict to a par_set
        par_set = np.array(par_set.values())[self._par_index]

        # Convert par_set to unit_space
        par_set = self._to_unit_space(par_set)

        # Evaluate the IMAGINE pipeline
        sims = self._img_pipe._get_observables(par_set)

        # Create empty dict of model_data
        mod_dict = {}

        # Loop over all observables
        for key, obs in sims.archive.items():
            # Convert coordinates into a format for PRISM
            coords_type, coords_list = self._convert_coords(obs.coords)

            # Take the average over all ensembles
            data = np.average(obs.data, axis=0)

            # Loop over all values with appropriate coordinates
            for coord, val in zip(coords_list, data):
                # Obtain data_idx
                idx = (*key, coords_type, *coord)

                # Add value
                mod_dict[idx] = val

        # Return mod_dict
        return(mod_dict)

    # Override get_md_var
    def get_md_var(self, *args, **kwargs):
        # TODO: Implement variance calculation to include random fields
        super().get_md_var(*args, **kwargs)
