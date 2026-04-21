"""
Módulo de logging para AstroPyFlow.

Configura logging estruturado com níveis INFO/DEBUG/WARNING/ERROR,
formato padronizado e handlers para console e arquivo.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configura e retorna um logger com handlers apropriados.

    Args:
        name: Nome do logger (geralmente __name__ do módulo).
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR).
        log_file: Caminho opcional para arquivo de log. Se None, apenas console.
        format_string: Formato customizado. Usa padrão se None.

    Returns:
        Logger configurado com handlers de console e arquivo (se especificado).

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Mensagem informativa")
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(funcName)s:%(lineno)d | %(message)s"
        )

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicação de handlers se logger já existir
    if logger.handlers:
        return logger

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo (se especificado)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger existente sem reconfigurar.

    Args:
        name: Nome do logger.

    Returns:
        Logger existente ou cria um novo básico.
    """
    return logging.getLogger(name)
