cristal.commons
===============

.. automodule:: cristal.commons
   :no-members:

Classes
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Class
     - Description
   * - :class:`cristal.commons.Distance <cristal.commons.distance.Distance>`
     - Class to compute the distance between each pair of two collections of inputs.
   * - :class:`cristal.commons.Incrementer <cristal.commons.incrementer.Incrementer>`
     - Class to increment the moment matrix :attr:`M` constructed from :attr:`N` points with new data points :attr:`X`.
   * - :class:`cristal.commons.Inverter <cristal.commons.inverter.Inverter>`
     - Class to invert a matrix :attr:`X` with optional regularization :attr:`eps`.
   * - :class:`cristal.commons.PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`
     - Class to generate vandermonde matrices for either 1D or nD data with the given :attr:`basis`.
   * - :class:`cristal.commons.Solver <cristal.commons.solver.Solver>`
     - Class to solve the equation :math:`z = v^T G^{-1} v`.
   * - :class:`cristal.commons.Storage <cristal.commons.storage.Storage>`
     - Class to loop over the data with a limited size to avoid OOM errors.
   * - :class:`cristal.commons.ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>`
     - Class to compute a threshold based on data properties.

Submodules / Subpackages
-----------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Module
     - Description
   * - :mod:`cristal.commons.base_commons`
     - Contains Base class for all commons classes. Useful to bind configuration dependencies to the class.
   * - :mod:`cristal.commons.distance`
     - Contains the :class:`Distance <cristal.commons.distance.Distance>` class used in static detectors.
   * - :mod:`cristal.commons.incrementer`
     - Contains the :class:`Incrementer <cristal.commons.incrementer.Incrementer>` class used in dynamic detectors.
   * - :mod:`cristal.commons.inverter`
     - Contains the :class:`Inverter <cristal.commons.inverter.Inverter>` class used in dynamic detectors.
   * - :mod:`cristal.commons.polynomial_basis`
     - Contains the :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>` class used in detectors.
   * - :mod:`cristal.commons.solver`
     - Contains the :class:`Solver <cristal.commons.solver.Solver>` class used in static detectors.
   * - :mod:`cristal.commons.storage`
     - Contains the :class:`Storage <cristal.commons.storage.Storage>` class used in detectors.
   * - :mod:`cristal.commons.threshold_scheme`
     - Contains the :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>` class used in detectors.

.. toctree::
   :hidden:

   cristal.commons.base_commons
   cristal.commons.distance
   cristal.commons.incrementer
   cristal.commons.inverter
   cristal.commons.polynomial_basis
   cristal.commons.solver
   cristal.commons.storage
   cristal.commons.threshold_scheme
