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
     - TypeVar(ndarray, Tensor)
   * - :const:`DTypeLike <cristal.types.DTypeLike>`
     - TypeVar(DTypeLike, dtype)
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
   * - :const:`TORCH_AVAILABLE <cristal.types.TORCH_AVAILABLE>`
     - bool(x) -> bool

Detailed reference
------------------

.. py:data:: ArrayLike
   :type: TypeVar
   :canonical: cristal.types.ArrayLike

   Constrained to ``ndarray``, ``Tensor``.

.. py:data:: DTypeLike
   :type: TypeVar
   :canonical: cristal.types.DTypeLike

   Constrained to ``DTypeLike``, ``dtype``.

.. automodule:: cristal.types
   :members:
   :member-order: groupwise
   :inherited-members:
   :show-inheritance:
   :special-members: __call__
   :private-members: _compute_scores
   :exclude-members: ArrayLike, DTypeLike
