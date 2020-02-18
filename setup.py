import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bretina",
    version="0.0.2",
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
        'Pillow>=6.2.1'
    ]
)
