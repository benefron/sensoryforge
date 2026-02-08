from setuptools import setup, find_packages

setup(
    name="sensoryforge",
    version="0.2.0",
    description="Modular, extensible framework for simulating sensory encoding across modalities",
    author="Sensory Forge Contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "torchaudio>=0.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "PyQt5>=5.15.0",
        "pyqtgraph>=0.12.0",
        "PyYAML>=6.0",
        "tqdm>=4.60",
        "matplotlib>=3.5",
    ],
    extras_require={
        "solvers": [
            "torchdiffeq>=0.2.3",
            "torchode>=0.2.0",
        ],
        "dsl": [
            "sympy>=1.11",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "mkdocs>=1.3.0",
            "mkdocs-material>=8.0.0",
            "mkdocs-include-markdown-plugin>=3.0.0",
        ],
    },
)
