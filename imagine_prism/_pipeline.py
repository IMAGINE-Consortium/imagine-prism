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
    """
    Defines a custom :class:`imagine.pipelines.Pipeline` class, the
    :class:`~PRISMPipeline` class.

    This class wraps a provided :class:`imagine.pipelines.Pipeline` class using
    the *PRISM* pipeline, and functions as both pipelines simultaneously.
    The internal *PRISM* pipeline can be accessed with the :attr:`~prism_pipe`
    attribute.
    Likelihood calculations in this class will perform *PRISM*'s hybrid
    sampling: Samples are first evaluated in the *PRISM* pipeline and only
    evaluated in the *IMAGINE* pipeline if the emulator flags this sample as
    plausible.
    For more information on hybrid sampling, see
    https://prism-tool.readthedocs.io/en/latest/user/using_prism.html#hybrid-sampling

    """

    # Class attributes
    overridden_attrs = ('_img_pipe', '_core_likelihood')

    # Override constructor
    def __init__(self, imagine_pipeline_obj, *args, **kwargs):
        """
        Initializes an instance of the :class:`~PRISMPipeline` class.

        Parameters
        ----------
        imagine_pipeline_obj : :obj:`imagine.pipelines.Pipeline` object
            The *IMAGINE* pipeline object that *PRISM* must wrap around.

        Optional
        --------
        args : positional arguments
            Positional arguments that must be provided to the constructor of
            the :class:`prism.Pipeline` class.
        kwargs : keyword arguments
            Keyword arguments that must be provided to the constructor of the
            :class:`prism.Pipeline` class.

        """

        # Save provided imagine_pipeline_obj
        self._img_pipe = imagine_pipeline_obj

        # Initialize IMAGINELink
        modellink_obj = IMAGINELink(self._img_pipe)

        # Initialize PRISM Pipeline
        self._prism_pipe = prism_Pipeline(modellink_obj, *args,
                                          prefix='imagine_', **kwargs)

        # Store PRISM's communicator
        self._prism_comm = self._prism_pipe._comm

        # Create a directory for storing runs for IMAGINE and set it
        run_dir = path.join(self._prism_pipe._working_dir, 'imagine_run')

        # Controller creates directory if necessary
        if self._prism_pipe._is_controller and not path.exists(run_dir):
            os.mkdir(run_dir)
        self._prism_comm.Barrier()

        # Set directories
        self._img_pipe.run_directory = run_dir
        self._img_pipe.chains_directory = None

        # Obtain the parameter names in IMAGINE
        par_names = self._img_pipe.get_par_names()

        # Obtain the sorting order IMAGINE->PRISM
        self._par_index = list(map(par_names.index, modellink_obj._par_name))

        # Set use_impl_prior to False
        self._use_impl_prior = False

    # Override method to use attributes from _img_pipe
    def __getattribute__(self, name):
        if(name not in PRISMPipeline.overridden_attrs
           and not name.startswith('__')
           and name in self._img_pipe.__dir__()):
            return(getattr(self._img_pipe, name))
        else:
            return(super().__getattribute__(name))

    # Override method to use attributes from _img_pipe
    def __setattr__(self, name, value):
        if(name not in PRISMPipeline.overridden_attrs
           and not name.startswith('__')
           and name in self._img_pipe.__dir__()):
            setattr(self._img_pipe, name, value)
        else:
            super().__setattr__(name, value)

    # Override method to use attributes from _img_pipe
    def __delattr__(self, name):
        if(name not in PRISMPipeline.overridden_attrs
           and not name.startswith('__')
           and name in self._img_pipe.__dir__()):
            delattr(self._img_pipe, name)
        else:
            super().__delattr__(name)

    # Override method to display attributes of both classes
    def __dir__(self):
        return(set(dir(self._img_pipe)).union(super().__dir__()))

    # %% CLASS PROPERTIES
    @property
    def prism_pipe(self):
        """
        :obj:`prism.Pipeline`: The *PRISM* pipeline object that is used in
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

        # If par_set is plausible, call likelihood method of _img_pipe
        if len(impl_sam):
            return(lnprior+self._img_pipe._core_likelihood(cube))

        # If par_set is not plausible, return -inf
        else:
            return(-np.infty)
