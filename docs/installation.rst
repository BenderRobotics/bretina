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

* `opencv-python`_ (>=4.1.1.0) - 
* `numpy`_ (>=1.17) - 
* `PIL`_ (>=5.1.2) -

.. _opencv-python: https://pypi.org/project/colorama/
.. _numpy: https://pypi.org/project/pyserial/
.. _PIL:   https://pypi.org/project/PyYAML/


External dependencies
~~~~~~~~~~~~~~~~~~~~~

Bretina uses OCR engine Tesseract for the optical character recognition. Tesseract can be downloaded from
https://github.com/tesseract-ocr/tesseract (tested with Tesseract version 5). After the installation, add tesseract.exe to your system ``PATH``.

For the Windows use installer provided by **Mannheim University Library**: https://github.com/UB-Mannheim/tesseract/wiki.
