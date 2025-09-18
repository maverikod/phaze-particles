#!/usr/bin/env python3
"""
Setup script for Phaze-Particles.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from setuptools import setup, find_packages

setup(
    name="phaze-particles",
    version="0.1.0",
    description="Elementary Particle Modeling Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vasiliy Zdanovskiy",
    author_email="vasilyvz@gmail.com",
    url="https://github.com/vasilyvz/phaze-particles",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phaze-particles=phaze_particles.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="physics particles modeling topological-solitons",
)
