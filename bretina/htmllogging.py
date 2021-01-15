from base64 import b64encode
import logging
import random

html_template_header = '''
<html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {
                background-color: #2a2a2a;
                font-size: 14px;
            }

            pre {
                vertical-align: top;
                margin: 0;
                padding: 0.05em 0.25em;
                white-space: pre-wrap;
                font-family: "Consolas", monospace;
            }

            .level--TRACE {
                color: #4F4F4F;
                font-weight: normal;
            }

            .level--DEBUG {
                color: #B5CDE3;
                font-weight: normal;
            }

            .level--INFO {
                color: #fff;
                font-weight: normal;
            }

            .level--WARNING {
                color: #ffc107;
                font-weight: normal;
            }

            .level--ERROR {
                color: #dc3545;
                font-weight: bold;
            }

            .level--FAIL {
                background-color: #FFDF81;
                color: #000;
                border: 1px solid #FFDF8173;
                border-radius: 0.25em;
                box-shadow: 0 0 0.33em #FFDF81;
                font-weight: bold;
            }

            .level--CRITICAL {
                background-color: #dc3545;
                color: #fff;
                border: 1px solid #dc354573;
                border-radius: 0.25em;
                box-shadow: 0 0 0.33em #dc3545;
                font-weight: bold;
            }

            .log.filter--trace .level--TRACE { display: none; }
            .log.filter--debug .level--DEBUG { display: none; }
            .log.filter--info .level--INFO { display: none; }
            .log.filter--warning .level--WARNING { display: none; }
            .log.filter--error .level--ERROR { display: none; }
            .log.filter--fail .level--FAIL { display: none; }
            .log.filter--critical .level--CRITICAL { display: none; }

            time {
                color: #6a6a6a;
                font-weight: normal;
            }

            /* Images ------------------------ */
            img {
                display: block;
                max-height: 180px;
                margin: 0.5em;
                transition: all 0.5s;
            }

            input[type="checkbox"].filter--image {
                display: none;
            }

            input[type="checkbox"].filter--image:checked + label img {
                max-height: 612px;
            }

            /* Log level filters ------------------------ */

            .filter {
                position: fixed;
                top: 2px;
                right: 2px;
                background-color: white;
                border: 2px solid #b3b3b3;
                border-radius: 2px;
                padding: 0.25em;
                font-family: sans-serif;
                font-weight: bold;
                text-align: center;
                opacity: 0.3;
                transition: all 0.25s;
            }

            .filter:hover{
                opacity: 1;
            }
        </style>
        <title></title>
    </head>
    <body>
        <div class="filter">
            <label><input type="checkbox" name="filter--trace" class="filter--trace" checked /> trace</label>
            <label><input type="checkbox" name="filter--debug" class="filter--level" checked /> debug</label>
            <label><input type="checkbox" name="filter--info" class="filter--level" checked /> info</label>
            <label><input type="checkbox" name="filter--warning" class="filter--level" checked /> warning</label>
            <label><input type="checkbox" name="filter--error" class="filter--level" checked /> error</label>
            <label><input type="checkbox" name="filter--fail" class="filter--level" checked /> fail</label>
            <label><input type="checkbox" name="filter--critical" class="filter--level" checked /> critical</label>
            <script>
                var filters = document.getElementsByClassName("filter--level");

                for (var i = 0; i < filters.length; i++)
                {
                    filters[i].addEventListener('change', function() {
                    const log = document.getElementById("log");

                    if (this.checked) {
                        log.classList.remove(this.name);
                    } else {
                        log.classList.add(this.name);
                    }
                    });
                }
            </script>
        </div>
        <div class="log" id="log" class="">
'''

html_template_footer = '</div></body></html>'
html_formatter = '<pre class="level--%(levelname)s"><time>%(asctime)s</time> %(levelname)s [%(module)s:%(funcName)s:%(lineno)d] %(message)s</pre>'
IMAGE_HEIGHT = 612


class HtmlHandler(logging.FileHandler):
    """
    File handler for the logging module with the HTML formated output allowing
    to embed OpenCV, PIL or PyPlot images into the log file.
    """

    def __init__(self, filename, mode='w', encoding="utf-8", delay=False):
        super().__init__(filename, mode, encoding, delay)
        fmt = logging.Formatter(html_formatter)
        self.setFormatter(fmt)

    def close(self):
        if self.stream:
            self.stream.writelines(html_template_footer)
        super().close()

    def _open(self):
        stream = super()._open()
        stream.writelines(html_template_header)
        return stream

    def emit(self, record):
        try:
            if isinstance(record.msg, ImageRecord):
                record.msg = record.msg.to_html()
            super().emit(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class ImageRecord(object):
    """
    Represents image to be logged to the HTML file.
    """
    _renderers = None

    def __init__(self, img=None, fmt="jpeg"):
        """
        :param img: image to be logged (OpenCV, PIL or PyPlot)
        :param fmt: image format to be used for coding ('jpeg' or 'png')
        """
        self.fmt = fmt
        self.img = img

    @classmethod
    def render(cls, img, fmt):
        """
        Tries to create renderers for OpenCV, PIL and PyPlot images.

        :param img: image to be logged (OpenCV, PIL or PyPlot)
        :param fmt: image format to be used for coding ('jpeg' or 'png')
        """
        if not cls._renderers:
            cls._renderers = []
            # Try to create OpenCV image renderer
            try:
                import cv2
                import numpy

                def render_opencv(img, fmt="png"):
                    if not isinstance(img, numpy.ndarray):
                        return None

                    if img.shape[0] > IMAGE_HEIGHT:
                        width = int(img.shape[1] * (IMAGE_HEIGHT / img.shape[0]))
                        img = cv2.resize(img.copy(), (width, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)

                    retval, buf = cv2.imencode(f".{fmt}", img)
                    if not retval:
                        return None

                    return buf, f"image/{fmt}"

                cls._renderers.append(render_opencv)
            except ImportError:
                pass

            # Try to create PIL image renderer
            try:
                from io import BytesIO
                from PIL import Image

                def render_pil(img, fmt="png"):
                    if not callable(getattr(img, "save", None)):
                        return None

                    output = BytesIO()
                    width, height = img.size

                    if height > IMAGE_HEIGHT:
                        width = int(width * (IMAGE_HEIGHT / height))
                        img = img.resize((width, IMAGE_HEIGHT))

                    # if format is jpeg, convert to RGB to remove transparency
                    if fmt in ("jpeg", "jpg"):
                        img = img.convert('RGB')

                    img.save(output, format=fmt)
                    contents = output.getvalue()
                    output.close()

                    return contents, f"image/{fmt}"

                cls._renderers.append(render_pil)
            except ImportError:
                pass

            # Try to create PyPlot image renderer
            try:
                from io import BytesIO

                def render_pyplot(img, fmt="png"):
                    if not callable(getattr(img, "savefig", None)):
                        return None

                    output = BytesIO()
                    img.savefig(output, format=fmt)
                    contents = output.getvalue()
                    output.close()

                    return contents, f"image/{fmt}"

                cls._renderers.append(render_pyplot)
            except ImportError:
                pass

        # Trying renderers we have one by one
        for renderer in cls._renderers:
            res = renderer(img, fmt)
            if res is not None:
                return res

        return None

    def __str__(self):
        return "[[ImageRecord]]"

    def to_html(self):
        """
        Converts image to HTML base64 representation.
        """
        res = self.render(self.img, self.fmt)

        if res is not None:
            data = b64encode(res[0]).decode()
            mime = res[1]
            _id = f'img-{id(data)}-{random.randint(0, 1000000)}'
            return f'<input type="checkbox" id="{_id}" class="filter--image"/><label for="{_id}"><img class="img" src="data:{mime};base64,{data}" /></label>'
        else:
            return f"<em>Rendering not supported for {self.fmt} | {repr(self.img)}.</em>"
