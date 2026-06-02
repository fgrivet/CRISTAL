cristal.core.detectors.dynamic
==============================

.. automodule:: cristal.core.detectors.dynamic
   :no-members:
   :no-index:

Classes
-------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Name
     - Description
   * - :class:`DyCF <cristal.core.detectors.dynamic.DyCF>`
     - Class to compute the original Christoffel function based outlier detection scores and predictions, adapted from :cite:t:`ducharlet2025leveraging`.
   * - :class:`DyCG <cristal.core.detectors.dynamic.DyCG>`
     - An extension of the :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>` class to detect anomalies based on the growth of the socres as the degree :attr:`n` increases.

Detailed reference
------------------

.. automodule:: cristal.core.detectors.dynamic
   :members:
   :show-inheritance:
   :special-members: __init__, __call__
   :private-members: _compute_scores
