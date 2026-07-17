cristal.core.detectors
======================

.. automodule:: cristal.core.detectors
   :no-members:

Classes
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Class
     - Description
   * - :class:`cristal.core.detectors.DyCF <cristal.core.detectors.dynamic.DyCF>`
     - Dynamical Christoffel Function detector.
   * - :class:`cristal.core.detectors.DyCG <cristal.core.detectors.dynamic.DyCG>`
     - Class to compute the original Christoffel function based outlier detection scores,
   * - :class:`cristal.core.detectors.KernelCF <cristal.core.detectors.kernel.KernelCF>`
     - Kernel-based Christoffel Function detector.
   * - :class:`cristal.core.detectors.KernelCG <cristal.core.detectors.kernel.KernelCG>`
     - Class to compute the kernel version of the Christoffel function based outlier detection scores,
   * - :class:`cristal.core.detectors.NeedleCF <cristal.core.detectors.needle.NeedleCF>`
     - Base class for the detectors.
   * - :class:`cristal.core.detectors.NeedleCG <cristal.core.detectors.needle.NeedleCG>`
     - An extension of the :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>` class
   * - :class:`cristal.core.detectors.MTSUCF <cristal.core.detectors.univariate.MTSUCF>`
     - Multivariate Time Series Univariate Christoffel Function detector.
   * - :class:`cristal.core.detectors.MTSUCG <cristal.core.detectors.univariate.MTSUCG>`
     - Class to compute our Univariate version of the Christoffel function based outlier detection scores for multivariate time series,
   * - :class:`cristal.core.detectors.UCF <cristal.core.detectors.univariate.UCF>`
     - Univariate Christoffel Function detector.
   * - :class:`cristal.core.detectors.UCG <cristal.core.detectors.univariate.UCG>`
     - Class to compute our Univariate version of the Christoffel function based outlier detection scores,

Submodules / Subpackages
-----------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Module
     - Description
   * - :mod:`cristal.core.detectors.base_detector`
     - Contains the class :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>`
   * - :mod:`cristal.core.detectors.dynamic`
     - Contains the original Christoffel function based outlier detection algorithm, adapted from :cite:t:`ducharlet2025leveraging`.
   * - :mod:`cristal.core.detectors.kernel`
     - Contains the kernel version of the Christoffel function based outlier detection algorithm, adapted from :cite:t:`askari2018kernel`.
   * - :mod:`cristal.core.detectors.needle`
     - Contains the Needle polynomial version of the Christoffel function based outlier detection algorithm,
   * - :mod:`cristal.core.detectors.univariate`
     - Contains our Univariate version of the Christoffel function based outlier detection algorithm.

.. toctree::
   :hidden:

   cristal.core.detectors.base_detector
   cristal.core.detectors.dynamic
   cristal.core.detectors.kernel
   cristal.core.detectors.needle
   cristal.core.detectors.univariate
