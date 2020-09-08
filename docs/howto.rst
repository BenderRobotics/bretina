.. _howto:

How To
======

.. _howto.structure:

Module structure
----------------

The Bretina package has to main parts:

* :ref:`Bretina module function set <bretina>` - set of generic funcitions for the image processing. These function can be used for the preprocessing of the images.
* :ref:`visualtestcase` - a support test case class inherited from the `unittest.TestCase <https://docs.python.org/3/library/unittest.html#unittest.TestCase>`_.
  `VisualTestCase` provides set of assertion methods, similar to the assertions of the `unittest`.

Start testing
-------------

First of all, get familiar with the unit testing in python. Bretina is based on the `unittest` framework, so start with https://docs.python.org/3/library/unittest.htm.


How to create a test
~~~~~~~~~~~~~~~~~~~~

Create a class inherited from the `bretina.VisualTestCase`. Bretina will import the `unittest` automatically.

.. code-block:: python

    import bretina      # no need to import unittest

    class Test_Icon(bretina.VisualTestCase):
        """
        This is the test case based on the VisualTest
        """

        def test_icon(self):
            """
            This is the test method
            """
            pass

During the `setUp` it is necessary to assign the camera object. `camera` property is required and used for the image acqusition.

.. code-block:: python

    def setUp(self):
        self.camera = Camera()

Camera object has to implement following inteface:

.. code-block:: python

    binaryImage = Camera::acquire_calibrated_image()
    binaryImage = Camera::acquire_calibrated_images(num_images:int, period:int)

See Brest `generic_camera` for reference (https://gitlab.benderrobotics.com/br/tools/brest/-/blob/devel/brest/cameras/generic_camera.py).

VisualTestCase takes image from the camera when the `capture` method is called. Usual flow of the test method is:

# set up test conditions,
# capture image with the `VisualTestCase.capture()` method,
# verify the image correctness.

The `capture` method stores the captured frame into the `VisualTestCase.img` property.

.. code-block:: python

    import bretina

    class Test_Icon(bretina.VisualTestCase):

        def setUp(self):
            self.camera = Camera()

        def test_icon(self):
            # set up test conditions

            self.capture()  # capture image with the camera

            # verify the image correctness with the assertions.

All the assertion method of the `unittest.TestCase` are supported. Upon that, additional set of assertions for the visual operations is provided by the `VisualTestCase`.

.. code-block:: python

    import bretina

    class Test_Icon(bretina.VisualTestCase):

        def setUp(self):
            self.camera = Camera()

        def test_icon(self):
            self.capture()
            self.assertImage((10, 10, 26, 26), "icon.png")      # assertion of the icon.png visible in the box (10, 10, 26, 26)


How to log a test results
~~~~~~~~~~~~~~~~~~~~~~~~~

To analyze failed test, bretina by default saves captured images of the failed assertions to the `log` directory. Alternative path can be set in `VisualTestCase.LOG_PATH` property.

In order to embed the image information into text log file, Bretina provides custom `logging` module handler allowing to create HTML log files with embedded images.

.. code-block:: python

    # create HTML handler
    hh = HtmlHandler(os.path.join(LOG_FOLDER, '{}.html'.format(LOG_FILE)), mode="w", encoding="utf-8")
    hh.setLevel(logging.DEBUG)
    logger.addHandler(hh)
