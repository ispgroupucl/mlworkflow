from setuptools import setup, find_packages
from os import path


setup(
    name="mlworkflow",
    author="Maxime Istasse",
    author_email="istassem@gmail.com",
    url="https://github.com/ispgroupucl/mlworkflow",
    license='MIT',
    version="0.1.0",
    python_requires='>=3.67',
    description="Helpers for Machine Learning experiments",
    long_description_content_type="text/markdown",
    packages=find_packages(include=("mlworkflow",)),
    install_requires=[],
)
