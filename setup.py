"""
setup.py — Package configuration for PyPI publishing.

Install in development mode:
    pip install -e .

Publish to PyPI:
    python setup.py sdist bdist_wheel
    twine upload dist/*
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="minigrad-framework",
    version="1.0.0",
    author="Tanishq Mangal",
    author_email="tanishkmangal3@gmail.com",
    description="A deep learning framework built from scratch in NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eternalcodertanishq3/minigrad",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "matplotlib>=3.6",
        ],
        "torch": [
            "torch>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "minigrad-train=examples.04_mnist_mlp:train",
        ],
    },
)
