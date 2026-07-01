cristal.types
=============

.. automodule:: cristal.types
   :no-members:
   :no-index:

Variables / Constants
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Name
     - Description
   * - :const:`ArrayLike <cristal.types.ArrayLike>`
     - The type of the main manipulated object.
   * - :const:`DTypeLike <cristal.types.DTypeLike>`
     - The data type of the main manipulated object.
   * - :const:`IMPLEMENTED_BACKEND <cristal.types.IMPLEMENTED_BACKEND>`
     - The implemented backends.
   * - :const:`IMPLEMENTED_DISTANCES <cristal.types.IMPLEMENTED_DISTANCES>`
     - The implemented distance metrics.
   * - :const:`IMPLEMENTED_INCREMENTERS <cristal.types.IMPLEMENTED_INCREMENTERS>`
     - The implemented incrementer methods.
   * - :const:`IMPLEMENTED_INVERTERS <cristal.types.IMPLEMENTED_INVERTERS>`
     - The implemented inverters methods.
   * - :const:`IMPLEMENTED_POLYNOMIAL_BASIS <cristal.types.IMPLEMENTED_POLYNOMIAL_BASIS>`
     - The implemented polynomial basis.
   * - :const:`IMPLEMENTED_SOLVERS <cristal.types.IMPLEMENTED_SOLVERS>`
     - The implemented solvers.
   * - :const:`IMPLEMENTED_STORAGES <cristal.types.IMPLEMENTED_STORAGES>`
     - The implemented storage methods.
   * - :const:`IMPLEMENTED_THRESHOLD_SCHEMES <cristal.types.IMPLEMENTED_THRESHOLD_SCHEMES>`
     - The implemented threshold schemes.

Detailed reference
------------------

.. py:data:: ArrayLike
   :type: TypeVar
   :canonical: cristal.types.ArrayLike

   The type of the main manipulated object.

   Constrained to ``ndarray``, ``Tensor``.

.. py:data:: DTypeLike
   :type: TypeVar
   :canonical: cristal.types.DTypeLike

   The data type of the main manipulated object.

   Constrained to ``Union``, ``dtype``.

.. automodule:: cristal.types
   :members:
   :show-inheritance:
   :special-members: __init__, __call__
   :private-members: _compute_scores
   :exclude-members: ArrayLike, DTypeLike
