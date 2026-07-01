cristal.config.detector\_config
===============================

.. automodule:: cristal.config.detector_config
   :no-members:
   :no-index:

Classes
-------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Name
     - Description
   * - :class:`DetectorConfig <cristal.config.detector_config.DetectorConfig>`
     - Class to store the backend and commons classes for all detectors.
   * - :class:`DynamicDetectorConfig <cristal.config.detector_config.DynamicDetectorConfig>`
     - Class to store the backend and commons classes for dynamic detectors: :class:`DyCF <cristal.core.detectors.dynamic.DyCF>`, :class:`DyCG <cristal.core.detectors.dynamic.DyCG>`, :class:`KernelCF <cristal.core.detectors.kernel.KernelCF>`, :class:`KernelCG <cristal.core.detectors.kernel.KernelCG>`.
   * - :class:`StaticDetectorConfig <cristal.config.detector_config.StaticDetectorConfig>`
     - Class to store the backend and commons classes for static detectors: :class:`UCF <cristal.core.detectors.univariate.UCF>`, :class:`UCG <cristal.core.detectors.univariate.UCG>`, :class:`NeedleCF <cristal.core.detectors.needle.NeedleCF>`, :class:`NeedleCG <cristal.core.detectors.needle.NeedleCG>`.

Variables / Constants
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Name
     - Description
   * - :const:`ConfigType <cristal.config.detector_config.ConfigType>`
     - The DetectorConfig type.

Detailed reference
------------------

.. py:data:: ConfigType
   :type: TypeVar
   :canonical: cristal.config.detector_config.ConfigType

   The DetectorConfig type.

   Bound to :class:`DetectorConfig <cristal.config.detector_config.DetectorConfig>`.

.. automodule:: cristal.config.detector_config
   :members:
   :show-inheritance:
   :special-members: __init__, __call__
   :private-members: _compute_scores
   :exclude-members: ConfigType
