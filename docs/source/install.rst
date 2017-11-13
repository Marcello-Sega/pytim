How to Install Pytim
********************

**IMPORTANT:** Pytim (mainly because of its dependency on MDAnalysis) is compatible with **Python 2.7 only**. Python 3.x is not supported.

There are three ways to install Pytim:

Download the source from github
-------------------------------


The package will download all the dependencies which are needed. Prerequisites for running the setup.py script are `setuptools` and `cython`.

.. code-block:: bash

	pip install setuptools --user  --upgrade
	pip install cython     --user  --upgrade

	git clone https://github.com/Marcello-Sega/pytim.git
	cd pytim
	python setup.py install --user

Note that on os-x it is usually better not to install Pytim system-wide using `sudo`, unless you know what you are doing, as newer versions of the operating system are protecting some system files to be overwritten even being super user.


Using pip to access the Python Package Index
--------------------------------------------

The package can also be installed directly from the Python Package Index, along with the prerequisites `setuptools` and `cython`.

.. code-block:: bash

	pip install setuptools --user  --upgrade
	pip install cython     --user  --upgrade

	pip install pytim      --user  --upgrade

Using Anaconda
--------------

Finally, Pytim is also available in the Anaconda Package Manager.

.. code-block:: bash

    conda install -c conda-forge pytim

....


.. toctree::

.. raw:: html
   :file: analytics.html

