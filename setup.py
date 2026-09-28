"""
setup.py — Package configuration for PyPI publishing.

Install in development mode:
    pip install -e .

Publish to PyPI:
    python setup.py sdist bdist_wheel
    twine upload dist/*
"""
import re
from pathlib import Path


def get_version() -> str:
    init_file = Path(__file__).parent / "minigrad" / "__init__.py"
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_file.read_text(encoding="utf-8"), re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in minigrad/__init__.py")
    return match.group(1)


if __name__ == "__main__":
    from setuptools import find_packages, setup

    readme_path = Path(__file__).parent / "README.md"
    long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    setup(
        name="minigrad-framework",
        version=get_version(),
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
                "ruff>=0.4",
                "mypy>=1.8",
            ],
            "torch": [
                "torch>=2.0",
            ],
        },
        entry_points={
            "console_scripts": [
                "minigrad=minigrad.cli:main",
                "minigrad-demo-scalar=minigrad.cli:demo_scalar",
                "minigrad-demo-linear=minigrad.cli:demo_linear_regression",
                "minigrad-demo-xor=minigrad.cli:demo_xor",
                "minigrad-bench-ops=minigrad.cli:bench_ops",
            ],
        },
    )
