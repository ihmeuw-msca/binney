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

.. code-block::

   pip install binney


You will also need to install :code:`ipopt`, which is an interior
point optimizer, with conda.

.. code-block::

   conda install -c conda-forge cyipopt


You can check to see if your installation worked correctly with :code:`pytest`.

.. code-block::

   # pip install pytest
   cd binney
   pytest


Documentation
-------------

For instructions on how to use :code:`binney`, see the API reference
for :class:`binney.run.run.BinneyRun`.

Quick Start
-----------

Here is a quick introduction to simulate data
and do a :code:`BinneyRun`. Please refer to
:class:`binney.run.run.BinneyRun` for documentation on
its arguments and methods.
We will simulate 500 observations from

.. math::

   x \sim Uniform(0, \pi) \\
   k \sim Binomial(n=100, p=p(x)) \\

where :math:`p(x) = \frac{e^{sin(x)}}{1 + e^{sin(x)}}`.
First, let's create the data frame representing this
simulation.

.. code-block::

   import matplotlib.pyplot as plt
   import pandas as pd
   import numpy as np

   from binney.run.run import BinneyRun

   np.random.seed(0)
   n = 500
   x = np.random.uniform(low=0, high=np.pi, size=n)
   p = np.exp(np.sin(x)) / (1 + np.exp(np.sin(x)))
   df = pd.DataFrame({
       'success': np.random.binomial(n=100, size=len(p), p=p),
       'total': np.repeat(100, repeats=len(p)),
       'p': p,
       'x': x
   })
   df.sort_values('x', inplace=True)
   df['p_hat'] = df['success'] / df['total']

Now, we can create the specifications for a binomial regression
model with :code:`binney`. If you wanted to include a shape
constraint on the spline, you would do so in the :code:`splines`
specifications below. See :class:`binney.run.run.BinneyRun`
for more details on these specs and documentation
about all of the arguments to the function.

.. code-block::

   splines = {
       'x': {
           'degree': 3,
           'knots_num': 4,
           'knots_type': 'frequency',
       }
   }
   b_run = BinneyRun(
       col_success='success',
       col_total='total',
       df=df,
       splines=splines,
       solver_method='ipopt',
       data_type='binomial'
   )


We can fit the model and create predictions from it.

.. code-block::

   b_run.fit()
   predictions = b_run.predict()


.. plot::

   import matplotlib.pyplot as plt
   import pandas as pd
   import numpy as np

   from binney.run.run import BinneyRun

   np.random.seed(0)
   n = 500
   x = np.random.uniform(low=0, high=np.pi, size=n)
   p = np.exp(np.sin(x)) / (1 + np.exp(np.sin(x)))
   df = pd.DataFrame({
       'success': np.random.binomial(n=100, size=len(p), p=p),
       'total': np.repeat(100, repeats=len(p)),
       'p': p,
       'x': x
   })
   df.sort_values('x', inplace=True)
   df['p_hat'] = df['success'] / df['total']

   splines = {
       'x': {
           'degree': 3,
           'knots_num': 4,
           'knots_type': 'frequency',
           'decreasing': False,
           'convex': False,
           'concave': False
       }
   }
   b_run = BinneyRun(
       col_success='success',
       col_total='total',
       df=df,
       splines=splines,
       solver_method='ipopt',
       data_type='binomial'
   )
   b_run.fit()
   predictions = b_run.predict()

   fig, ax = plt.subplots(1, 2, figsize=(8, 4))

   ax[0].plot(df.x, df.p, color='blue')
   ax[0].scatter(df.x, df.p_hat, color='red')
   ax[0].set_title("True p with sample data.")
   ax[0].set_xlabel("x")
   ax[0].set_ylabel("p")

   ax[1].plot(df.x, df.p, color='blue')
   ax[1].plot(df.x, predictions, color='red')
   ax[1].set_title("True v. fitted p")
   ax[1].set_xlabel("x")
   ax[1].set_ylabel("p")

You can then create uncertainty as well by doing:

.. code-block::

   b_run.make_uncertainty(n_boots=50)
   draws = b_run.predict_draws(df=df)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_reference/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
