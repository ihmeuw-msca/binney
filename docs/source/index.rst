.. binney documentation master file, created by
   sphinx-quickstart on Wed Aug  5 13:38:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to binney's documentation!
===================================

:code:`binney` is a package for doing binomial regression
(including logistic regression) with b-splines! See below
for installation instructions.

Installation
------------

You can install :code:`binney` with :code:`pip`:

```
pip install binney
```

You will also need to install :code:`ipopt`, which is an interior
point optimizer, with conda.

```
conda install -c conda-forge cyipopt
```

You can check to see if your installation worked correctly with :code:`pytest`.

```
# pip install pytest
cd binney
pytest
```

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_reference/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
