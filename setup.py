from setuptools import find_namespace_packages, setup
import pathlib

with open(
    pathlib.Path(__file__).absolute().parent / "README.md", "r"
) as fh:
    long_description = fh.read()

setup(
    name="rikai-ocr",
    version="0.0.1",
    license="Apache License, Version 2.0",
    author="Darcy Shen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/da-tubi/rikai-ocr",
    python_requires=">=3.7",
    install_requires=["rikai>=0.1.6"],
    extras_require={
        "dev": [
            "black",
            "isort",
            # for testing
            "pytest",
        ]
    },
    packages=find_namespace_packages(include=["rikai.*"]),
)
