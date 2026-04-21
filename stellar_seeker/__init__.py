"""
StellarSeeker - Análise de Curvas de Luz e Descoberta de Exoplanetas

Uma aplicação Python completa para análise de curvas de luz astronômicas
e descoberta de exoplanetas utilizando dados públicos da missão TESS.

Autor: StellarSeeker Development Team
Licença: MIT
"""

__version__ = "1.0.0"
__author__ = "StellarSeeker Development Team"

from .core import StellarSeeker
from .utils import (
    tic_to_coordinates,
    btjd_to_datetime,
    format_period
)

__all__ = [
    'StellarSeeker',
    'tic_to_coordinates',
    'btjd_to_datetime',
    'format_period'
]
