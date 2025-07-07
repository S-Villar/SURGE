#!/usr/bin/env python
"""
Setup script for SURGE - Surrogate Unified Robust Generation Engine
"""

from setuptools import setup, find_packages
import os

# Read the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="surge-surrogate",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="Surrogate Unified Robust Generation Engine - A modular AI/ML framework for building fast, accurate, and uncertainty-aware surrogate models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Álvaro Sánchez Villar",
    author_email="your.email@example.com",
    url="https://github.com/your-username/SURGE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "jupyter",
            "matplotlib",
            "pandas",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "numpydoc",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
