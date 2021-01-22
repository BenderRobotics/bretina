import setuptools
import re

# Load description from the README
with open("README.md", "r") as fh:
    long_description = fh.read()

# Load version from teh module
with open("bretina/__init__.py", 'r', encoding='utf-8') as fh:
    content = fh.read()
    reg = re.compile(r"__version__\s*=\s*['\"]([\d\.]+)['\"]")
    version = reg.search(content).group(1)
    print(version)

setuptools.setup(
    name="bretina",
    version=version,
    author="Bender Robotics",
    author_email="kumpan@benderrobotics.com",
    description="Bender Robotics Visual Test Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.benderrobotics.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'opencv-python>=4.1.1.0',
        'numpy>=1.17',
        'pytesseract>=0.3.0',
        'Pillow>=6.2.1',
        'htmllogging @ https://pypi.benderrobotics.com/packages/htmllogging-0.0.1.tar.gz#md5=6e7c2a6ccaf4e2b7701c1be849f0fb64'
    ]
)
