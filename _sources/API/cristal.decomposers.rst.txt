cristal.decomposers
===================

.. automodule:: cristal.decomposers
   :exclude-members: BaseDecomposer, BaseWindowDecomposer, FourierDecomposer, WaveletDecomposer, WindowFourierDecomposer, WindowWaveletDecomposer


Modules
-------

.. autosummary::
   :toctree:
   :recursive:

   base
   fourier
   wavelet
   window

Classes
-------

.. autosummary::

   IMPLEMENTED_DECOMPOSERS
   base.BaseDecomposer
   fourier.FourierDecomposer
   fourier.WindowFourierDecomposer
   wavelet.WaveletDecomposer
   wavelet.WindowWaveletDecomposer

Functions
---------

.. autosummary::

   base.BaseDecomposer.decompose
   base.BaseDecomposer.full_decompose
   base.BaseDecomposer.get_coefficients_at_indices
   base.BaseDecomposer.plot
   base.BaseDecomposer.reconstruct
   base.BaseDecomposer.transform