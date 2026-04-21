"""
AstroPyFlow - Aplicação Python modular para Astronomia.

Este pacote fornece ferramentas para:
- Coleta e validação de dados astronômicos
- Visualização com Matplotlib + Astropy
- Cálculos de distância, luminosidade e conversões
- Simulações orbitais simplificadas
- Classificação básica de exoplanetas
"""

__version__ = "0.1.0"
__author__ = "AstroPyFlow Team"
__email__ = "contact@astropyflow.dev"

from astropyflow.utils.logger import setup_logger

# Configurar logger global ao importar o pacote
logger = setup_logger("astropyflow")

__all__ = ["__version__", "__author__", "__email__"]
