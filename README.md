# Bretina

Bretina is a support for the visual-based tests.

## Building Bretina from the source
If you want to build Bretina from the source, you need to clone the repository first.
Then checkout to `devel` branch to get the latest version, or to `feature/*` branches for the cutting edge versions.

    $ git checkout devel

For building the python wheel, we use GNU **make**.

**make** expects that your pip3 installation will be available under `pip` command. Also there is a possibility that `setup.py` may fail
on `bdist_wheel` as an not known argument. To fix this, install `wheel` package again.

Navigate to top-level directory of the cloned repository and you will be able to use the following commands:

    $ make install    # First time Bretina installation. It will build the source and install it using pip
    $ make reinstall  # Pulled new commit? Use this to build new wheel, uninstall and install the new version using pip
    $ make all        # Just builds the wheel
    $ make clean      # Cleans the build directories and files

Builded wheel is located in `Bretina/dist/`.

## Installing make
On Ubuntu like machines execute the following command:

    $ sudo apt install make

If you are on windows machine, you can install a [MinGW](http://www.mingw.org/). In the MinGW installer choose `mingw32-base-bin` and
`msys-base-bin` packages, then click on *Installation* and *Apply changes*.
Don't forget to add the **make** binary to the system `PATH`. Default install location should be `C:\MinGW\msys\1.0\bin\`.
