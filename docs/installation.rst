.. _installation:

Installation
============

.. _installation.download-and-install-latest-release:

Download and install latest release
------------------------------------

.. code-block:: sh

    $ pip install bretina

.. _installation.python-version:

Python version
--------------

Bretina supports Python 3.6 and above on Windows and Linux

.. _installation.python-dependencies:

Dependencies
-------------------

These dependencies will be installed automatically when installing Bretina

* `opencv-python`_ (>=4.1.1.0) - OpenCV packages,
* `numpy`_ (>=1.17) - package for array computing,
* `Pillow`_ (>=7.2.0) - the Python Imaging Library.

.. _opencv-python: https://pypi.org/project/opencv-python/
.. _numpy: https://pypi.org/project/numpy/
.. _Pillow: https://pypi.org/project/Pillow/


External dependencies
~~~~~~~~~~~~~~~~~~~~~

Bretina uses OCR engine Tesseract for the optical character recognition. Tesseract can be downloaded from
https://github.com/tesseract-ocr/tesseract (tested with Tesseract version 5). After the installation, add tesseract.exe to your system ``PATH``.

For the Windows use installer provided by **Mannheim University Library**: https://github.com/UB-Mannheim/tesseract/wiki.
