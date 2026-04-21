"""
Handler de dados astronômicos.

Implementa funções para:
- Carregamento de dados de fontes públicas (CSV, APIs)
- Validação e sanitização de catálogos estelares
- Criação de dados mock para testes e demonstrações
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from astropyflow.utils.validators import DataValidator

logger = logging.getLogger(__name__)


def load_star_data(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Carrega dados estelares de arquivo CSV, FITS ou outro formato suportado.

    Args:
        file_path: Caminho para o arquivo de dados.
        file_format: Formato do arquivo ('csv', 'fits', etc.). Se None, detecta automaticamente.
        **kwargs: Argumentos adicionais para pandas.read_csv ou astropy.table.Table.read.

    Returns:
        DataFrame com dados carregados.

    Raises:
        FileNotFoundError: Se arquivo não existir.
        ValueError: Se formato não for suportado.

    Example:
        >>> df = load_star_data("catalog.csv", delimiter=",")
        >>> print(df.columns)
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"Arquivo não encontrado: {file_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    if file_format is None:
        file_format = path.suffix.lower().replace(".", "")

    try:
        if file_format == "csv":
            df = pd.read_csv(path, **kwargs)
            logger.info(f"Carregado CSV: {len(df)} linhas, {len(df.columns)} colunas")
        elif file_format == "fits":
            table = Table.read(path, **kwargs)
            df = table.to_pandas()
            logger.info(f"Carregado FITS: {len(df)} linhas, {len(df.columns)} colunas")
        elif file_format in ["json", "parquet", "excel"]:
            df = getattr(pd, f"read_{file_format}")(path, **kwargs)
            logger.info(f"Carregado {file_format.upper()}: {len(df)} linhas")
        else:
            # Tenta CSV como fallback
            logger.warning(f"Formato '{file_format}' não reconhecido. Tentando CSV...")
            df = pd.read_csv(path, **kwargs)

        return df

    except Exception as e:
        logger.error(f"Erro ao carregar dados de {file_path}: {e}")
        raise


def create_mock_catalog(n_stars: int = 100, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Cria um catálogo estelar simulado para testes e demonstrações.

    Gera dados realistas incluindo:
    - Coordenadas (RA, Dec) em graus
    - Magnitude aparente e absoluta
    - Distância em parsecs
    - Tipo espectral
    - Temperatura efetiva
    - Raio em raios solares

    Args:
        n_stars: Número de estrelas no catálogo.
        seed: Seed para reprodutibilidade do random.

    Returns:
        DataFrame com catálogo simulado.

    Example:
        >>> catalog = create_mock_catalog(50, seed=42)
        >>> catalog.head()
    """
    if seed is not None:
        np.random.seed(seed)

    # Gerar coordenadas aleatórias (distribuição uniforme na esfera)
    ra = np.random.uniform(0, 360, n_stars)  # Ascensão reta em graus
    dec = np.random.uniform(-90, 90, n_stars)  # Declinação em graus

    # Gerar distâncias (mais estrelas próximas, menos distantes)
    distance_pc = np.random.exponential(scale=100, size=n_stars) + 10  # Mínimo 10 pc

    # Tipos espectrais com frequências aproximadas da realidade
    spectral_types = ["O", "B", "A", "F", "G", "K", "M"]
    spectral_weights = [0.00003, 0.13, 0.6, 3, 7.6, 12.1, 76]  # Aproximado
    spectral_classes = np.random.choice(
        spectral_types, size=n_stars, p=np.array(spectral_weights) / sum(spectral_weights)
    )

    # Temperaturas baseadas no tipo espectral (K)
    temp_ranges = {
        "O": (30000, 50000),
        "B": (10000, 30000),
        "A": (7500, 10000),
        "F": (6000, 7500),
        "G": (5200, 6000),
        "K": (3700, 5200),
        "M": (2400, 3700),
    }

    temperature = np.array([
        np.random.uniform(*temp_ranges[st]) for st in spectral_classes
    ])

    # Raios em raios solares (correlacionado com tipo espectral)
    radius_ranges = {
        "O": (6.0, 15.0),
        "B": (2.5, 6.0),
        "A": (1.4, 2.5),
        "F": (1.15, 1.4),
        "G": (0.96, 1.15),
        "K": (0.7, 0.96),
        "M": (0.1, 0.7),
    }

    radius = np.array([
        np.random.uniform(*radius_ranges[st]) for st in spectral_classes
    ])

    # Magnitude absoluta baseada em temperatura e raio (simplificado)
    # M_bol = M_bol,Sun - 2.5 * log10(L/L_sun)
    # L/L_sun = (R/R_sun)^2 * (T/T_sun)^4
    t_sun = 5778  # K
    luminosity_ratio = (radius ** 2) * (temperature / t_sun) ** 4
    m_bol_sun = 4.74
    absolute_mag = m_bol_sun - 2.5 * np.log10(luminosity_ratio)

    # Magnitude aparente: m = M + 5*log10(d/10)
    apparent_mag = absolute_mag + 5 * np.log10(distance_pc / 10)

    # Adicionar algum ruído realista
    apparent_mag += np.random.normal(0, 0.1, n_stars)

    # Criar DataFrame
    data = {
        "star_id": [f"STAR_{i:04d}" for i in range(n_stars)],
        "ra": ra,
        "dec": dec,
        "distance_pc": distance_pc,
        "apparent_mag": apparent_mag,
        "absolute_mag": absolute_mag,
        "spectral_class": spectral_classes,
        "temperature_K": temperature,
        "radius_solar": radius,
        "luminosity_solar": luminosity_ratio,
    }

    # Adicionar alguns valores nulos aleatórios para testar validação (~5%)
    n_nulls = int(n_stars * 0.05)
    null_indices = np.random.choice(n_stars, n_nulls, replace=False)
    data["temperature_K"][null_indices[:n_nulls//2]] = np.nan

    df = pd.DataFrame(data)

    logger.info(f"Criado catálogo mock: {n_stars} estrelas")
    return df


def validate_catalog(
    data: pd.DataFrame,
    strict_mode: bool = False,
) -> Dict[str, Any]:
    """
    Valida um catálogo de dados astronômicos.

    Executa verificações completas:
    - Valores nulos
    - Outliers
    - Faixas válidas para grandezas astronômicas
    - Unidades consistentes

    Args:
        data: DataFrame com dados do catálogo.
        strict_mode: Se True, falha em qualquer problema. Se False, apenas alerta.

    Returns:
        Dicionário com resultados da validação e dados limpos se aplicável.

    Example:
        >>> result = validate_catalog(df, strict_mode=True)
        >>> if result['is_valid']:
        ...     clean_data = result['cleaned_data']
    """
    validator = DataValidator(tolerance_sigma=5.0, min_valid_ratio=0.90)

    # Definir unidades esperadas
    expected_units = {
        "ra": u.deg,
        "dec": u.deg,
        "distance_pc": u.pc,
        "temperature_K": u.K,
    }

    # Definir faixas válidas
    valid_ranges = {
        "ra": (0, 360),
        "dec": (-90, 90),
        "distance_pc": (0.01, 100000),  # 0.01 a 100 kpc
        "apparent_mag": (-10, 30),  # Magnitudes típicas
        "absolute_mag": (-10, 20),
        "temperature_K": (1000, 100000),
        "radius_solar": (0.01, 1000),
    }

    # Executar validação completa
    validation_result = validator.validate_full(
        data,
        expected_units=expected_units,
        valid_ranges=valid_ranges,
    )

    # Preparar dados limpos
    cleaned_data = data.copy()

    # Remover linhas com nulos críticos
    critical_cols = ["ra", "dec", "distance_pc"]
    null_mask = cleaned_data[critical_cols].isnull().any(axis=1)
    if null_mask.any():
        logger.info(f"Removendo {null_mask.sum()} linhas com nulos críticos")
        cleaned_data = cleaned_data[~null_mask]

    # Filtrar outliers extremos (>7σ)
    if len(cleaned_data) > 10:
        for col in ["distance_pc", "temperature_K", "radius_solar"]:
            if col in cleaned_data.columns:
                series = cleaned_data[col].dropna()
                if len(series) > 10:
                    mean = series.mean()
                    std = series.std()
                    if std > 0:
                        extreme_mask = np.abs(cleaned_data[col] - mean) > 7 * std
                        if extreme_mask.any():
                            logger.info(f"Removendo {extreme_mask.sum()} outliers extremos em {col}")
                            cleaned_data = cleaned_data[~extreme_mask]

    # Resetar índices após limpeza
    cleaned_data = cleaned_data.reset_index(drop=True)

    result = {
        "is_valid": validation_result["summary"]["is_valid"],
        "validation_details": validation_result,
        "cleaned_data": cleaned_data,
        "original_rows": len(data),
        "cleaned_rows": len(cleaned_data),
        "rows_removed": len(data) - len(cleaned_data),
    }

    if strict_mode and not result["is_valid"]:
        logger.error("Validação falhou em modo estrito")
        raise ValueError("Catálogo não passou na validação estrita")

    logger.info(
        f"Validação concluída: {result['cleaned_rows']}/{result['original_rows']} linhas válidas"
    )

    return result


def convert_coordinates_to_skycoord(
    data: pd.DataFrame,
    ra_col: str = "ra",
    dec_col: str = "dec",
    frame: str = "icrs",
) -> SkyCoord:
    """
    Converte colunas de coordenadas para objeto SkyCoord do Astropy.

    Args:
        data: DataFrame com colunas de RA e Dec.
        ra_col: Nome da coluna de ascensão reta (graus).
        dec_col: Nome da coluna de declinação (graus).
        frame: Sistema de referência (default: 'icrs').

    Returns:
        SkyCoord com todas as coordenadas.

    Example:
        >>> coords = convert_coordinates_to_skycoord(df)
        >>> coords[0]
    """
    if ra_col not in data.columns or dec_col not in data.columns:
        logger.error(f"Colunas {ra_col} ou {dec_col} não encontradas")
        raise KeyError(f"Colunas necessárias: {ra_col}, {dec_col}")

    # Remover valores nulos para conversão
    mask = data[ra_col].notna() & data[dec_col].notna()
    ra_values = data.loc[mask, ra_col].values
    dec_values = data.loc[mask, dec_col].values

    coords = SkyCoord(
        ra=ra_values * u.deg,
        dec=dec_values * u.deg,
        frame=frame,
    )

    logger.info(f"Criado SkyCoord com {len(coords)} coordenadas")
    return coords


def export_catalog(
    data: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "csv",
    overwrite: bool = False,
) -> Path:
    """
    Exporta catálogo para arquivo.

    Args:
        data: DataFrame para exportar.
        output_path: Caminho do arquivo de saída.
        format: Formato de saída ('csv', 'fits', 'parquet').
        overwrite: Se True, sobrescreve arquivo existente.

    Returns:
        Path do arquivo criado.

    Raises:
        FileExistsError: Se arquivo já existe e overwrite=False.
    """
    path = Path(output_path)

    if path.exists() and not overwrite:
        logger.error(f"Arquivo já existe: {path}")
        raise FileExistsError(f"Arquivo já existe: {path}. Use overwrite=True.")

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "csv":
            data.to_csv(path, index=False)
        elif format == "fits":
            table = Table.from_pandas(data)
            table.write(path, overwrite=overwrite)
        elif format == "parquet":
            data.to_parquet(path, index=False)
        else:
            raise ValueError(f"Formato não suportado: {format}")

        logger.info(f"Catálogo exportado para {path} ({len(data)} linhas)")
        return path

    except Exception as e:
        logger.error(f"Erro ao exportar catálogo: {e}")
        raise
