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
     - Class to store the backend and commons classes for dynamic detectors:
   * - :class:`StaticDetectorConfig <cristal.config.detector_config.StaticDetectorConfig>`
     - Class to store the backend and commons classes for static detectors:

Variables / Constants
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Name
     - Description
   * - :const:`ConfigType <cristal.config.detector_config.ConfigType>`
     - The DetectorConfig type.
   * - :const:`TORCH_AVAILABLE <cristal.config.detector_config.TORCH_AVAILABLE>`
     - bool(x) -> bool

Detailed reference
------------------

.. py:data:: ConfigType
   :type: TypeVar
   :canonical: cristal.config.detector_config.ConfigType

   The DetectorConfig type.

   Bound to :class:`DetectorConfig <cristal.config.detector_config.DetectorConfig>`.

.. automodule:: cristal.config.detector_config
   :members:
   :member-order: groupwise
   :inherited-members:
   :show-inheritance:
   :special-members: __call__
   :private-members: _compute_scores
   :exclude-members: ConfigType
