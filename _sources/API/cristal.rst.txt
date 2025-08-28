CRISTAL
=======

.. automodule:: cristal
   :exclude-members: BaggingDyCF, DyCF, DyCFPlotter, DyCG, UTSCF, IMPLEMENTED_DECOMPOSERS, IMPLEMENTED_INCREMENTERS, IMPLEMENTED_INVERTERS, IMPLEMENTED_POLYNOMIALS, IMPLEMENTED_REGULARIZERS, MomentsMatrix, MultivariatePolynomialBasis, decomposers, detectors, evaluators, incrementers, inverters, moments_matrix, plotters, polynomials, regularizers, type_checking


Subpackages
-----------


.. autosummary::

   decomposers
   detectors
   evaluation
   incrementers
   inverters
   moments_matrix
   plotters
   polynomials
   regularizers


Modules
-------

.. autosummary::

   type_checking

Classes
-------

The following classes are available using directly ``cristal.ClassName`` instead of ``cristal.package.module.ClassName``.

Implemented options
^^^^^^^^^^^^^^^^^^^

.. autosummary::

   decomposers.IMPLEMENTED_DECOMPOSERS
   incrementers.IMPLEMENTED_INCREMENTERS
   inverters.IMPLEMENTED_INVERTERS
   polynomials.IMPLEMENTED_POLYNOMIALS
   regularizers.IMPLEMENTED_REGULARIZERS


Detectors
^^^^^^^^^

.. autosummary::

   detectors.datastreams.BaggingDyCF
   detectors.datastreams.DyCF
   detectors.datastreams.DyCG
   detectors.timeseries.UTSCF


Utils
^^^^^

.. autosummary::

   moments_matrix.moments_matrix.MomentsMatrix
   polynomials.base.MultivariatePolynomialBasis
   plotters.dycf.DyCFPlotter


.. toctree::
   :maxdepth: 3
   :hidden:

   cristal.decomposers
   cristal.detectors
   cristal.evaluation
   cristal.incrementers
   cristal.inverters
   cristal.moments_matrix
   cristal.plotters
   cristal.polynomials
   cristal.regularizers
   cristal.type_checking