"""
Módulo core do AstroPyFlow.

Contém funcionalidades principais de:
- data_handler: Coleta, validação e sanitização de dados
- calculations: Cálculos de distância, luminosidade e conversões
"""

from astropyflow.core.data_handler import load_star_data, validate_catalog, create_mock_catalog
from astropyflow.core.calculations import (
    calculate_angular_distance,
    calculate_linear_distance,
    calculate_luminosity,
    calculate_absolute_magnitude,
    convert_units,
)

__all__ = [
    "load_star_data",
    "validate_catalog",
    "create_mock_catalog",
    "calculate_angular_distance",
    "calculate_linear_distance",
    "calculate_luminosity",
    "calculate_absolute_magnitude",
    "convert_units",
]
