import os
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f.readlines()]

with open("README.md") as r:
    readme = r.read()

setup(
    name="graph-measures",
    version="0.1.57",
    license="GPL",
    author="Itay Levinas",
    maintainer="Ziv Naim",
    maintainer_email="zivnaim3@gmail.com",
    url="https://github.com/louzounlab/graph-measures",
    description="A python package for calculating topological graph features on cpu/gpu",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=["gpu", "graph", "topological-features-calculator"],
    license_files=["LICENSE"],
    install_requires=requirements,
    packages=find_packages('.'),
    python_requires=">=3.6.8",
    include_package_data=True,
    package_data={'': ['*.pkl']},
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: C++',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
    ],
)