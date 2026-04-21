"""
Setup script para instalação do StellarSeeker.

Use: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stellar-seeker",
    version="1.0.0",
    author="StellarSeeker Development Team",
    description="Análise de Curvas de Luz e Descoberta de Exoplanetas com Dados do TESS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stellar-seeker/stellar-seeker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "stellar-seeker=examples.demo_examples:main",
        ],
    },
)
