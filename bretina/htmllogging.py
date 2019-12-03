from base64 import b64encode
import logging
import time

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

            .level--DEBUG {
                color: #6a6a6a;
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

            .level--CRITICAL {
                background-color: #dc3545;
                color: #ffffff;
                border: 1px solid #dc354573;
                border-radius: 0.25em;
                box-shadow: 0 0 0.33em #dc3545;
                font-weight: bold;
            }

            time {
                color: #6a6a6a;
                font-weight: normal;
            }

            img {
                display: block;
                max-height: 180px;
                margin: 0.5em;
                transition: all 0.5s;
            }

            input[type="checkbox"] {
                display: none;
            }

            input[type="checkbox"]:checked + label img {
                max-height: 544px;
            }
        </style>
        <title></title>
    </head>
    <body>
'''

html_template_footer = '</body></html>'

html_formatter = '<pre class="level--%(levelname)s"><time>%(asctime)s</time> %(levelname)s [%(module)s:%(funcName)s:%(lineno)d] %(message)s</pre>'


class HtmlHandler(logging.FileHandler):
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


class ImageRecord(object):
    _renderers = None

    def __init__(self, img=None, fmt="jpeg"):
        self.fmt = fmt
        self.img = img

    @classmethod
    def render(cls, img, fmt):
        """
        Tries to create renderers for OpenCV, PIL and PyPlot images.
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
        res = self.render(self.img, self.fmt)

        if res is not None:
            data = b64encode(res[0]).decode()
            mime = res[1]
            _id = f'img-{id(data)}-{time.thread_time_ns()}'
            return f'<input type="checkbox" id="{_id}" /><label for="{_id}"><img class="img" src="data:{mime};base64,{data}" /></label>'
        else:
            return f"<em>Rendering not supported for {self.fmt} | {repr(self.img)}.</em>"
