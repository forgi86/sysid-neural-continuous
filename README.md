# Continuous-time system identification with neural networks 

<!--- This repository contains the Python code to reproduce the results of the 
paper "Continuous-time system identification with neural networks" by Marco Forgione and Dario Piga. --->

The following fitting methods for neural dynamical models are implemented and tested

 1. Full simulation error minimization
 2. Truncated simulation error minimization
 3. Soft-constrained integration
 4. One-step prediction error minimization


# Folders:
* [torchid](torchid):  PyTorch implementation several neural dynamical models
* [examples](examples): examples of neural dynamical models identification 
* [common](common): definition of metrics R-square, RMSE, fit index 

Three [examples](examples) are presented:

* [RLC](examples/RLC): A nonlinear series RLC circuit. Simulated dataset generated by our python code
* [CTS](examples/CTS): Cascaded Tanks System. Experimental dataset from http://www.nonlinearbenchmark.org
* [EMPS](examples/EMPS): Electro-Mechanical Positioning System. Experimental dataset from http://www.nonlinearbenchmark.org

For the [RLC](examples/RLC) example, the main scripts are:

 *  ``RLC_fit_truncated.py``: identification with truncated simulation error minimization
 *  ``RLC_fit_full``: identification with full simulation error minimization
 *  ``RLC_fit_1step``: identification with one-step prediction error minimization
 *  ``RLC_fit_soft.py``: identification with soft-constrained integration
 *  ``RLC_eval_sim.py``: Evaluate simulation performance of the identified models
 *  ``RLC_OE_comparison.m``: Linear Output Error identification in Matlab (``oe`` method)
 *  ``RLC_subspace_comparison.m``: Linear subspace identification in Matlab (``n4sid`` method)
  
Similar scripts are provided for the other examples.

# Software requirements:
Simulations were performed on a Python 3.7 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * sympy
 * pytorch (version 1.4)
 * numba
 * nodepy
 * tensorboard
 
These dependencies may be installed through the commands:

```
conda install numpy numba scipy sympy pandas matplotlib ipython
conda install pytorch torchvision cpuonly -c pytorch
pip install tensorboard nodepy
```
