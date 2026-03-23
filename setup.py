"""
setup.py
========
Package definition for the Quant Alpha Research Platform.

Purpose
-------
This file makes the entire `quant_alpha` codebase an installable Python package.
This is a cornerstone of institutional-grade software engineering, providing
several critical benefits over a simple script-based approach.

Usage
-----
The package is typically installed in "editable" mode during development. This
links the installed package directly to the source code, so any changes are
immediately reflected without needing to reinstall.

    # Install in editable mode for development
    $ pip install -e .

    # Install development and testing dependencies
    $ pip install -e .[dev]

    # Standard installation (for deployment or distribution)
    $ pip install .

Importance
----------
1.  **Dependency Management**: Centralizes all project dependencies, ensuring
    reproducible environments across different machines and for CI/CD pipelines.
    Eliminates the need for manual `pip install -r requirements.txt`.
2.  **Namespace Integrity**: Once installed, the `quant_alpha` package is available
    globally in the Python environment, completely removing the need for fragile
    `sys.path` manipulations in individual scripts. This prevents a wide class
    of `ImportError` bugs.
3.  **Tooling Integration**: Enables standard Python tools like `pytest` to
    automatically discover the source code and tests without path configuration.
4.  **Distribution**: Provides a standard way to build and distribute the
    platform as a wheel (`.whl`) or source distribution (`.tar.gz`).

Tools & Frameworks
------------------
*   **`setuptools`**: The standard library for defining Python packages. `setup()`
    is the main entry point, and `find_packages()` automatically discovers
    the source code directories to include.
"""

from setuptools import setup, find_packages

setup(
    name="quant_alpha",
    version="0.1.0",
    description="Institutional-Grade Quantitative Equity Alpha Platform",
    author="Anurag Patkar",
    # Automatically find the 'quant_alpha' package to include.
    packages=find_packages(include=["quant_alpha", "quant_alpha.*"]),
    python_requires=">=3.9",
    # Core runtime dependencies required for the platform to function.
    # Version pinning (e.g., >=) ensures a stable base while allowing minor updates.
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
        "catboost>=1.2.0",
        "numba>=0.57.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "yfinance>=0.2.0",
        "optuna>=3.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "python-dotenv>=1.0.0",
        "python-dateutil>=2.8.2",
        "lxml>=4.9.0",
        "pyarrow>=10.0.0",
        "hmmlearn>=0.3.0",
        "cvxpy>=1.3.0",
        "pandas_market_calendars>=4.1.0",
        "statsmodels>=0.13.0",
        "streamlit>=1.20.0",
    ],
    # Optional dependencies for development, testing, and code quality.
    # Installed via `pip install .[dev]`. This keeps the core production
    # environment lean.
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "fastparquet>=2023.10.0",  # Fallback engine for parquet operations
            "types-python-dateutil",  # For mypy dateutil stubs
            "types-requests",         # For mypy requests stubs
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)