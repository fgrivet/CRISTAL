cristal.backend
===============

.. automodule:: cristal.backend
   :no-members:

Classes
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Class
     - Description
   * - :class:`cristal.backend.NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>`
     - Backend using Numpy and Scipy.
   * - :class:`cristal.backend.TorchBackend <cristal.backend.torch_backend.TorchBackend>`
     - Backend using torch with `cpu` and `GPU` implementations.

Submodules / Subpackages
-----------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Module
     - Description
   * - :mod:`cristal.backend.base_backend`
     - Contains Base class for all backends.
   * - :mod:`cristal.backend.keops_backend`
     - TODO
   * - :mod:`cristal.backend.numpy_backend`
     - Contains the :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>`: a backend using Numpy and Scipy.
   * - :mod:`cristal.backend.torch_backend`
     - Contains the :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>`: a backend using torch with `cpu` and `GPU` implementations.

.. toctree::
   :hidden:

   cristal.backend.base_backend
   cristal.backend.keops_backend
   cristal.backend.numpy_backend
   cristal.backend.torch_backend
