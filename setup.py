import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bretina",
    version="0.0.1",
    author="Bender Robotics",
    author_email="kumpan@benderrobotics.com",
    description="Bender Robotics visual test support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.benderrobotics.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
