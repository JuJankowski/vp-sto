from __future__ import absolute_import  # now local imports must use .
# big difference between PY2 and PY3:
from __future__ import division
from __future__ import print_function
# only necessary for python 2.5 (not supported) and not in heavy use
from __future__ import with_statement
___author__ = "Julius Jankowski and Lara Brudermüller"
__license__ = "BSD 3-clause"

import warnings as _warnings

# __package__ = 'vpsto'
try:
    import numpy
    del numpy
except ImportError:
    _warnings.warn('Install `numpy` ("pip install numpy").')
else:
    from . import obf, vpsto
    from .vpsto import VPSTO

del division, print_function, absolute_import, with_statement

__version__ = "1.0.0"
