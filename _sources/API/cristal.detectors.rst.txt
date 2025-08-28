cristal.detectors
=================

.. automodule:: cristal.detectors
   :exclude-members: BaseDetector, BaggingDyCF, DyCF, DyCG, UTSCF


Modules
-------

.. autosummary::
   :toctree:
   :recursive:

   base
   datastreams
   timeseries

Classes
-------

.. autosummary::

   base.BaseDetector
   datastreams.BaggingDyCF
   datastreams.DyCF
   datastreams.DyCG
   timeseries.UTSCF


Functions
---------

.. autosummary::

   base.BaseDetector.assert_shape_fitted
   base.BaseDetector.assert_shape_unfitted
   base.BaseDetector.copy
   base.BaseDetector.decision_function
   base.BaseDetector.eval_update
   base.BaseDetector.fit
   base.BaseDetector.fit_predict
   base.BaseDetector.is_fitted
   base.BaseDetector.load_model
   base.BaseDetector.method_name
   base.BaseDetector.predict
   base.BaseDetector.predict_update
   base.BaseDetector.save_model
   base.BaseDetector.score_samples
   base.BaseDetector.update