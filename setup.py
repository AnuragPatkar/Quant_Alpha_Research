from setuptools import setup, find_packages

setup(
    name="quant_alpha",
    version="0.1.0",
    description="Institutional-Grade Quantitative Equity Alpha Platform",
    author="Anurag Patkar",
    packages=find_packages(include=["quant_alpha", "quant_alpha.*"]),
    python_requires=">=3.9",
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
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)