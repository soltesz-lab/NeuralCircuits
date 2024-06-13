# NeuralCircuits

Small biophysical neural circuit models


## Prerequisites

1) **Numpy** 

The standard python module for matrix and vector computations: https://pypi.python.org/pypi/numpy.

2) **Scipy** 

The standard python module for statistical analysis: http://www.scipy.org/install.html.

3) **Matplotlib**

The standard python module for data visualization: http://matplotlib.org/users/installing.html.

4) **NEURON**

A simulator for biophysical models of neurons and networks of neurons: https://github.com/neuronsimulator/nrn

## Building Model

Once NEURON is installed, set the `PATH` and `PYTHONPATH ` environment variables as follows:

```
export PYTHONPATH=$HOME/install/lib/python:$PYTHONPATH
export PATH=$HOME/install/bin:$PATH
```

As in a typical NEURON workflow, use `nrnivmodl` to translate MOD files:

```
nrnivmodl mechanisms
```

## Running Simulations
