.. _bretina:

.. automodule:: bretina
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::

.. autofunction:: color

Bretina is internally using the OpenCV way of the representation color as ``(B, G, R)`` tuple in range of ``0 - 255``. However, when function takes color
as an argument, all four color representations are posible:

1. ``(B, G, R)`` tuple (e.g. orange ``(0, 165, 255)``),
2. string hexadecimal code (e.g. orange ``#ffa500``,
3. string shorthand hexadecimal code - abbreviates 6-character RRGGBB colors into 3-character RGB (e.g. orange ``#ff6600`` can be written as ``#f60``) or
4. string name based on CSS color names pallete keywords (https://www.w3.org/TR/css-color-3/#svg-color).

.. autofunction:: color_str

.. autofunction:: dominant_colors

    .. image:: _static/fig_dominant_colors_1.png
        :alt: dominant colors figure
        :scale: 50%

    .. code-block:: python

        bretina.dominant_colors(img, n=3)

    .. raw:: html

        <p>gives</p>
        <p>
        <ul>
            <li><span style="background-color: rgb(41, 128, 184);" class="color-block"></span>(184, 128, 41)</li>
            <li><span style="background-color: rgb(79, 80, 80);" class="color-block"></span>(80, 80, 79)</li>
            <li><span style="background-color: rgb(224, 224, 225);" class="color-block"></span>(225, 224, 224)</li>
        </ul>
        </p>

    Increasing number of colors

    .. code-block:: python

        bretina.dominant_colors(img, n=5)

    .. raw:: html

        <p>provides better resolution of grey shades:</p>
        <p>
        <ul>
            <li><span style="background-color: rgb(41, 128, 184);" class="color-block"></span>(184, 128, 41)</li>
            <li><span style="background-color: rgb(115, 116, 117);" class="color-block"></span>(117, 116, 115)</li>
            <li><span style="background-color: rgb(43,  44,  44);" class="color-block"></span>(44, 44, 43)</li>
            <li><span style="background-color: rgb(254,  255,  255);" class="color-block"></span>(255, 255, 254)</li>
            <li><span style="background-color: rgb(185, 183, 180);" class="color-block"></span>(185, 183, 180)</li>
        </ul>
        </p>

.. autofunction:: dominant_color

.. autofunction:: active_color

    This function is usefull to determine color of the object.
    Returned color is the major color in the image which is not the background color.
    The background color can be set, or defined automatically from the image border area.

    .. image:: _static/fig_active_color_1.png
        :alt: active colors figure
        :scale: 50%

    .. code-block:: python

        bretina.active_color(img)

    .. raw:: html

        <p>gives <span style="background-color: rgb(41, 128, 184);" class="color-block"></span>(184, 128, 41)</p>

.. autofunction:: mean_color

.. autofunction:: background_color

.. autofunction:: background_lightness

.. autofunction:: color_std

.. autofunction:: lightness_std

.. autofunction:: rgb_distance

.. autofunction:: rgb_rms_distance

.. autofunction:: hue_distance

.. autofunction:: lightness_distance

.. autofunction:: lab_distance

.. autofunction:: ab_distance

    Distance between green and red color can be expressed as Euclidian distance in plane of a-b coordinates.

    .. image:: _static/fig_ab_distance_1.png
        :alt: active colors figure
        :scale: 80%

.. autofunction:: draw_border

    .. image:: _static/fig_draw_border_1.png
        :alt: active colors figure
        :scale: 50%

.. autofunction:: img_to_grayscale

.. autofunction:: text_rows

    5 rows of the text would be detected in the following image.

    .. image:: _static/fig_text_rows_1.png
        :alt: text rows figure
        :scale: 50%

.. autofunction:: text_cols

    2 cols of the text would be detected in the following image.

    .. image:: _static/fig_text_cols_1.png
        :alt: text cols figure
        :scale: 50%

.. autofunction:: split_cols

.. autofunction:: gamma_calibration

.. autofunction:: adjust_gamma

.. autofunction:: crop

.. autofunction:: read_text

    OCR engine of tesseract works well only for the dark text on the white background. ``read_text`` will try to automatically gather lightness of the background and inverte image with dark background. However, if the text background is apriori known, it is usefull to pass this information as ``bgcolor`` parameter.

    ``floodfill`` option is usefull to unify text background:

    .. image:: _static/fig_floodfill_1.png
        :alt: text rows figure
        :scale: 25%

    .. image:: _static/fig_floodfill_2.png
        :alt: text rows figure
        :scale: 25%

    .. image:: _static/fig_floodfill_3.png
        :alt: text rows figure
        :scale: 25%

.. autofunction:: img_diff

.. autofunction:: resize

.. autofunction:: recognize_animation

.. autofunction:: separate_animation_template

.. autofunction:: format_diff

.. autofunction:: compare_str

.. autofunction:: remove_accents

.. autofunction:: normalize_lang_name

.. autofunction:: color_region_detection

.. autofunction:: img_hist_diff

.. autofunction:: merge_images

.. autofunction:: _blank_image
