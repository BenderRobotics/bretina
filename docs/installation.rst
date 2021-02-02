.. _installation:

Installation
============

.. _installation.download-and-install-latest-release:

Download and install latest release
------------------------------------

Do not forget to install external dependencies!
.. code-block:: sh

    $ pip install bretina

.. _installation.python-version:

Dependencies
-------------------

Camera hardware
~~~~~~~~~~~~~~~

Most common use of the Bretina is together with a camera capturing the device.
Camera configuration, control and the image pre-processing are not part of the
Bretina. You can use camera provided by *Brest*, or your own custom implementation.

Bretina was developed with the [FLIR Blackfly](https://www.flir.com/products/blackfly-s-usb3/)
camera, but any other industrial grade camera should work fine. Webcams may be
used as well, but the image quality is usually worse and therefore the performace
may not be optimal.

External dependencies
~~~~~~~~~~~~~~~~~~~~~

If you want to verify correctness of the text it is nessesary to install OCR engine Tesseract for the optical character
recognition.
Bretina uses OCR engine Tesseract for the optical character recognition. Tesseract can be downloaded from
https://github.com/tesseract-ocr/tesseract (tested with Tesseract version 5). After the installation, add location of
the ``tesseract.exe`` to your system ``PATH``.

For the Windows use installer provided by **Mannheim University Library**:
https://github.com/UB-Mannheim/tesseract/wiki.

For the best OCR performance install the slower, but more accurate datasets ``tessdata_best``
(https://github.com/tesseract-ocr/tessdata_best). Extract the
downloaded archive into the installation directory of the tesseract OCR.

This is an expected structure of the tesseract installation directory:

+ *Tesseract-OCR* - tesseract installation
    * *tessdata* - original tessdata dataset
        + *afr.traineddata*
        + ...
    * *tessdata_best* - extracted best dataset
        + *afr.traineddata*
        + ...

Python dependencies
~~~~~~~~~~~~~~~~~~~~~

Bretina supports Python 3.6 and above on Windows and Linux

.. _installation.python-dependencies:

These dependencies will be installed automatically when installing Bretina

* `opencv-python`_ (>=4.1.1.0) - OpenCV packages,
* `numpy`_ (>=1.17) - package for array computing,
* `Pillow`_ (>=7.2.0) - the Python Imaging Library.

.. _opencv-python: https://pypi.org/project/opencv-python/
.. _numpy: https://pypi.org/project/numpy/
.. _Pillow: https://pypi.org/project/Pillow/
