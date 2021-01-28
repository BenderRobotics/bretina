# Bretina - Python Image Testing Framework

Bretina is a python package designed to support testing of the images.

![Intro](docs/_static/fig_intro.png)

Bretina is designed as an extension of the python
[unit test](https://docs.python.org/3/library/unittest.html) module. Provides
set of assertions which can be used to verify correctness of the image.

In typical application, content of the device LCD display is captured with a
camera and Bretina is used to verify correctness of the defined regions in the
image - such as region color, presence of an icon, correctness of the text
representation and other aspects.

## Documentation

can be found on https://docs.benderrobotics.com/bretina (Not publicly available ATM).

## Download and install latest release

Bretina can be downloaded from within the internal BR network by calling:

```console
    $ pip install bretina
```

Bretina uses OCR engine Tesseract for the optical character recognition.
Tesseract has to be installed as a standalone application and registered into
system `PATH`. Installation files can be downloaded from
https://github.com/tesseract-ocr/tesseract (tested with Tesseract version 5).
Windows installer is provided by **Mannheim University Library** at
https://github.com/UB-Mannheim/tesseract/wiki.

After the installation, add path to the `tesseract.exe` to your system `PATH`.

For the best OCR performance install the slower, but more accurate datasets
`tessdata_best` (https://github.com/tesseract-ocr/tessdata_best). In the
Tesseract installation directory, remove directory `tessdata`, place extracted
`tessdata_best` and rename to `tessdata`.

## Building Bretina from the source

If you want to build Bretina from the source, you need to clone the repository first.
Then checkout to `devel` branch to get the latest version, or to `feature/*` branches for the cutting edge versions.

```console
    $ git checkout devel
```

For building the python wheel, we use GNU **make**.

**make** expects that your pip3 installation will be available under `pip`
command. Also there is a possibility that `setup.py` may fail on `bdist_wheel`
as an not known argument. To fix this, install `wheel` package again.

Navigate to top-level directory of the cloned repository and you will be able to
use the following commands:

```console
    $ make install    # First time Bretina installation. It will build the source and install it using pip
    $ make reinstall  # Pulled new commit? Use this to build new wheel, uninstall and install the new version using pip
    $ make all        # Just builds the wheel
    $ make clean      # Cleans the build directories and files
```

Builded wheel is located in `Bretina/dist/`.

## Installing make

On Ubuntu like machines execute the following command:

```console
    $ sudo apt install make
```

If you are on windows machine, you can install a [MinGW](http://www.mingw.org/). In the MinGW installer choose `mingw32-base-bin` and
`msys-base-bin` packages, then click on *Installation* and *Apply changes*.
Don't forget to add the **make** binary to the system `PATH`. Default install location should be `C:\MinGW\msys\1.0\bin\`.
