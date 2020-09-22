# -*- coding: utf-8 -*-


# %% IMPORTS
# Built-in imports
import os
from os import path

# Package imports
from imagine.pipelines import Pipeline as imagine_Pipeline
import numpy as np
from prism import Pipeline as prism_Pipeline

# IMAGINE-PRISM imports
from imagine_prism.modellink import IMAGINELink

# All declaration
__all__ = ['PRISMPipeline']


# %% CLASS DEFINITIONS
class PRISMPipeline(imagine_Pipeline):
    # Class attributes
    overridden_attrs = ('_img_pipe', '_core_likelihood')

    # Override constructor
    def __init__(self, imagine_pipeline_obj, *args, **kwargs):
        # Save provided imagine_pipeline_obj
        self._img_pipe = imagine_pipeline_obj

        # Initialize IMAGINELink
        modellink_obj = IMAGINELink(self._img_pipe)

        # Initialize PRISM Pipeline
        self._prism_pipe = prism_Pipeline(modellink_obj, *args,
                                          prefix='imagine_', **kwargs)

        # Store PRISM's communicator
        self._prism_comm = self._prism_pipe._comm

        # Create a directory for storing chains for IMAGINE and set it
        chain_dir = path.join(self._prism_pipe._working_dir, 'imagine_chains')

        # Controller creates directory if necessary
        if self._prism_pipe._is_controller and not path.exists(chain_dir):
            os.mkdir(chain_dir)
        self._prism_comm.Barrier()

        # Set directory
        self._img_pipe.chains_directory = chain_dir

        # Obtain the parameter names in IMAGINE
        par_names = self._img_pipe.get_par_names()

        # Obtain the sorting order IMAGINE->PRISM
        self._par_index = list(map(par_names.index, modellink_obj._par_name))

        # Set use_impl_prior to False
        self._use_impl_prior = False

    # If requested attribute is not a method, use comm for getattr
    def __getattribute__(self, name):
        if(name not in PRISMPipeline.overridden_attrs and
           not name.startswith('__') and
           name in self._img_pipe.__dir__()):
            return(getattr(self._img_pipe, name))
        else:
            return(super().__getattribute__(name))

    # If requested attribute is not a method, use comm for setattr
    def __setattr__(self, name, value):
        if(name not in PRISMPipeline.overridden_attrs
           and not name.startswith('__') and
           name in self._img_pipe.__dir__()):
            setattr(self._img_pipe, name, value)
        else:
            super().__setattr__(name, value)

    # If requested attribute is not a method, use comm for delattr
    def __delattr__(self, name):
        if(name not in PRISMPipeline.overridden_attrs
           and not name.startswith('__') and
           name in self._img_pipe.__dir__()):
            delattr(self._img_pipe, name)
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

    # Override _core_likelihood
    def _core_likelihood(self, cube):
        # Convert provided cube to proper parameter set for PRISM
        sam = np.array(cube[self._par_index], ndmin=2)
        sam = self._prism_pipe._modellink._to_par_space(sam)

        # Check if par_set is within parameter space and return -inf if not
        par_rng = self._prism_pipe._modellink._par_rng
        if not ((par_rng[:, 0] <= sam[0])*(sam[0] <= par_rng[:, 1])).all():
            return(-np.infty)

        # Obtain the current emul_i
        emul_i = self._prism_pipe._emulator._emul_i

        # If it is zero, then impl_sam is always true
        if not emul_i:
            impl_sam = [True]
            lnprior = 0

        # Else, check what sampling is requested and analyze par_set
        elif self._use_impl_prior:
            impl_sam, lnprior = self._prism_pipe._make_call(
                '_evaluate_sam_set', emul_i, sam, 'hybrid')
        else:
            impl_sam = self._prism_pipe._make_call(
                '_evaluate_sam_set', emul_i, sam, 'analyze')
            lnprior = 0

        # If par_set is plausible, call super method
        if len(impl_sam):
            return(lnprior+super()._core_likelihood(cube))

        # If par_set is not plausible, return -inf
        else:
            return(-np.infty)
