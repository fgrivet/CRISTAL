cristal.core
============

.. automodule:: cristal.core
   :no-members:

Classes
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Class
     - Description
   * - :class:`cristal.core.UCF <cristal.core.detectors.univariate.UCF>`
     - Class to compute our Univariate version of the Christoffel function based outlier detection algorithm.
   * - :class:`cristal.core.UCG <cristal.core.detectors.univariate.UCG>`
     - An extension of the :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>` class to detect anomalies based on the growth of the socres as the degree :attr:`n` increases.
   * - :class:`cristal.core.DyCF <cristal.core.detectors.dynamic.DyCF>`
     - Class to compute the original Christoffel function based outlier detection scores and predictions, adapted from :cite:t:`ducharlet2025leveraging`.
   * - :class:`cristal.core.DyCG <cristal.core.detectors.dynamic.DyCG>`
     - An extension of the :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>` class to detect anomalies based on the growth of the socres as the degree :attr:`n` increases.
   * - :class:`cristal.core.KernelCF <cristal.core.detectors.kernel.KernelCF>`
     - Class to compute the kernel version of the Christoffel function based outlier detection scores and predictions, adapted from :cite:t:`askari2018kernel`.
   * - :class:`cristal.core.KernelCG <cristal.core.detectors.kernel.KernelCG>`
     - An extension of the :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>` class to detect anomalies based on the growth of the socres as the degree :attr:`n` increases.
   * - :class:`cristal.core.NeedleCF <cristal.core.detectors.needle.NeedleCF>`
     - Base class for the detectors.
   * - :class:`cristal.core.NeedleCG <cristal.core.detectors.needle.NeedleCG>`
     - An extension of the :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>` class to detect anomalies based on the growth of the socres as the degree :attr:`n` increases.

Submodules / Subpackages
-----------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Module
     - Description
   * - :mod:`cristal.core.detectors`
     - Contains the Christoffel-based detectors.

.. toctree::
   :hidden:

   cristal.core.detectors
