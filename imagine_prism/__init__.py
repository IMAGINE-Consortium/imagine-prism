# -*- coding: utf-8 -*-

"""
IMAGINE-PRISM

"""


# %% IMPORTS AND DECLARATIONS
# Submodule imports
from .__version__ import __version__
from ._pipeline import PRISMPipeline

# Subpackage imports
from . import modellink
from .modellink import *

# All declaration
__all__ = ['PRISMPipeline', 'modellink']
__all__.extend(modellink.__all__)

# Author declaration
__author__ = "Ellert van der Velden (@1313e)"
