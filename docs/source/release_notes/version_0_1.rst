Version 0.1 (2026-07-17)
========================

.. include:: legend.rst


Version 0.0.3
-------------
:mod:`Backend <cristal.backend>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- |Add| Functions :func:`Backend.make_windows() <cristal.backend.base_backend.Backend.make_windows>`, :func:`Backend.add_at() <cristal.backend.base_backend.Backend.add_at>`, and :func:`Backend.divide() <cristal.backend.base_backend.Backend.divide>`

:mod:`Commons <cristal.commons>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- |Change| Add parameter :const:`z` to :func:`PolynomialBasis.make_v() <cristal.commons.polynomial_basis.PolynomialBasis.make_v>` for univariate detectors to estimate the density of the original measure :math:`\mu`

:mod:`Configuration <cristal.config>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- |Change| Automatically change the default :class:`Backend <cristal.backend.base_backend.Backend>` from :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>` to :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>` if `torch` is not installed
- |Fix| Make binding work with a Pipeline of :class:`BasePreprocessor <cristal.preprocessing.base_preprocessor.BasePreprocessor>`

:mod:`Detectors <cristal.core.detectors>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- |Change| Add parameter :const:`z` for univariate detectors to estimate the density of the original measure :math:`\mu`
- |Add| Function :func:`BaseDetector.decision_function() <cristal.core.detectors.base_detector.BaseDetector.decision_function>` for compatibility with PyOD and sklearn
- |Fix| Function :func:`BaseDetector.score_samples() <cristal.core.detectors.base_detector.BaseDetector.score_samples>` to correctly assign scores of the reduced dimension last batch

:mod:`Preprocessing <cristal.preprocessing>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- |Add| :class:`BasePreprocessor <cristal.preprocessing.base_preprocessor.BasePreprocessor>`, :class:`MinMaxScaler <cristal.preprocessing.scalers.minmax.MinMaxScaler>`, and :class:`Windowizer <cristal.preprocessing.windowing.window.Windowizer>`

Other
~~~~~
- |Improve| Documentation
- |Improve| Tests

Bug score UCF

Version 0.0.2
-------------
:mod:`Detectors <cristal.core.detectors>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- |Add| First version of the analysis of multivariate time series with :class:`MTSUCF <cristal.core.detectors.univariate.MTSUCF>` and :class:`MTSUCG <cristal.core.detectors.univariate.MTSUCG>`
- |Change| Add parameters :const:`online` and :const:`quantile` to :func:`BaseDetector.score_samples() <cristal.core.detectors.base_detector.BaseDetector.score_samples>`

Other
~~~~~
- |Improve| Documentation
- |Improve| Tests


Version 0.0.1
-------------
- |Add| First release of CRISTAL