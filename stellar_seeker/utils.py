"""
Funções utilitárias para o StellarSeeker.

Módulo com funções auxiliares para conversão de coordenadas,
datas e formatação de períodos.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import lightkurve as lk

logger = logging.getLogger(__name__)


def tic_to_coordinates(tic_id: int) -> Optional[Tuple[float, float]]:
    """
    Converte um identificador TIC em coordenadas celestes (RA/Dec).
    
    Esta função consulta o catálogo TESS através do lightkurve para obter
    as coordenadas equatoriais (Ascensão Reta e Declinação) de uma estrela.
    
    Args:
        tic_id (int): Identificador TIC da estrela.
        
    Returns:
        tuple or None: Tupla (ra, dec) em graus, ou None se não encontrado.
                      - ra: Ascensão Reta em graus (0-360)
                      - dec: Declinação em graus (-90 a +90)
                      
    Example:
        >>> coords = tic_to_coordinates(261136679)
        >>> if coords:
        ...     ra, dec = coords
        ...     print(f"RA: {ra:.4f}°, Dec: {dec:.4f}°")
    """
    logger.info(f"Buscando coordenadas para TIC {tic_id}")
    
    try:
        # Buscar informações do alvo
        search_result = lk.search_lightcurve(f"TIC {tic_id}", author='SPOC')
        
        if len(search_result) == 0:
            logger.warning(f"Nenhuma informação encontrada para TIC {tic_id}")
            return None
        
        # Obter coordenadas do primeiro resultado
        row = search_result[0]
        
        # Tentar obter RA e Dec diretamente
        if hasattr(row, 'ra') and hasattr(row, 'dec'):
            ra = float(row.ra)
            dec = float(row.dec)
            logger.info(f"Coordenadas encontradas: RA={ra:.4f}°, Dec={dec:.4f}°")
            return (ra, dec)
        
        # Alternativa: tentar do target_name
        # Se não houver coordenadas diretas, retornar None
        logger.warning("Coordenadas não disponíveis nos metadados")
        return None
        
    except Exception as e:
        logger.error(f"Erro ao buscar coordenadas: {str(e)}")
        return None


def btjd_to_datetime(btjd_value: float) -> datetime:
    """
    Converte data Juliana Baricêntrica TESS (BTJD) em objeto datetime.
    
    BTJD é definido como JD - 2457000.0, onde JD é o Julian Date.
    Esta conversão permite manipulação temporal intuitiva em Python.
    
    Args:
        btjd_value (float): Valor BTJD a ser convertido.
        
    Returns:
        datetime: Objeto datetime do Python correspondente.
        
    Example:
        >>> dt = btjd_to_datetime(1400.5)
        >>> print(f"Data: {dt.strftime('%Y-%m-%d %H:%M')}")
    """
    # BTJD = JD - 2457000.0
    jd_value = btjd_value + 2457000.0
    
    # Converter JD para datetime
    time_obj = Time(jd_value, format='jd')
    dt = time_obj.to_datetime()
    
    return dt


def format_period(days: float) -> str:
    """
    Formata um período em dias para string legível.
    
    Converte um valor de período em dias para uma string formatada
    mostrando dias, horas, minutos e segundos.
    
    Args:
        days (float): Período em dias.
        
    Returns:
        str: String formatada como "X dias, Y horas, Z minutos, W segundos".
        
    Example:
        >>> period_str = format_period(6.26)
        >>> print(period_str)
        '6 dias, 6 horas, 14 minutos, 24 segundos'
        
        >>> format_period(0.5)
        '0 dias, 12 horas, 0 minutos, 0 segundos'
    """
    if days < 0:
        raise ValueError("Período deve ser positivo")
    
    # Calcular componentes
    total_seconds = days * 24 * 3600
    
    d = int(total_seconds // (24 * 3600))
    remaining = total_seconds % (24 * 3600)
    
    h = int(remaining // 3600)
    remaining = remaining % 3600
    
    m = int(remaining // 60)
    s = int(remaining % 60)
    
    # Construir string
    parts = []
    if d > 0:
        parts.append(f"{d} dia{'s' if d != 1 else ''}")
    if h > 0 or d > 0:
        parts.append(f"{h} hora{'s' if h != 1 else ''}")
    if m > 0 or h > 0 or d > 0:
        parts.append(f"{m} minuto{'s' if m != 1 else ''}")
    parts.append(f"{s} segundo{'s' if s != 1 else ''}")
    
    return ", ".join(parts)


def get_target_info(tic_id: int) -> Optional[dict]:
    """
    Obtém informações completas sobre um alvo TIC.
    
    Função utilitária que combina coordenadas, magnitude e outras
    informações disponíveis sobre uma estrela do catálogo TESS.
    
    Args:
        tic_id (int): Identificador TIC da estrela.
        
    Returns:
        dict or None: Dicionário com informações do alvo, ou None se não encontrado.
        
    Example:
        >>> info = get_target_info(261136679)
        >>> if info:
        ...     print(f"Estela: TIC {info['tic_id']}")
        ...     print(f"Magnitude TESS: {info.get('magnitude', 'N/A')}")
    """
    logger.info(f"Obtendo informações para TIC {tic_id}")
    
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", author='SPOC')
        
        if len(search_result) == 0:
            return None
        
        row = search_result[0]
        
        info = {
            'tic_id': tic_id,
            'target_name': getattr(row, 'target_name', f'TIC {tic_id}'),
            'sector': getattr(row, 'sector', None),
            'mission': getattr(row, 'mission', 'TESS'),
            'distance': getattr(row, 'distance', None),
        }
        
        # Adicionar coordenadas se disponíveis
        coords = tic_to_coordinates(tic_id)
        if coords:
            info['ra'] = coords[0]
            info['dec'] = coords[1]
        
        return info
        
    except Exception as e:
        logger.error(f"Erro ao obter informações: {str(e)}")
        return None


def calculate_signal_to_noise(flux_values: list) -> float:
    """
    Calcula a relação sinal-ruído (SNR) de uma série de fluxos.
    
    Args:
        flux_values (list): Lista de valores de fluxo.
        
    Returns:
        float: Relação sinal-ruído calculada.
    """
    import numpy as np
    
    flux_array = np.array(flux_values)
    mean_flux = np.mean(flux_array)
    std_flux = np.std(flux_array)
    
    if std_flux == 0:
        return float('inf')
    
    return mean_flux / std_flux


def normalize_flux(flux_values: list, method: str = 'median') -> list:
    """
    Normaliza uma série de fluxos usando diferentes métodos.
    
    Args:
        flux_values (list): Lista de valores de fluxo.
        method (str): Método de normalização ('median', 'mean', 'minmax').
        
    Returns:
        list: Fluxos normalizados.
    """
    import numpy as np
    
    flux_array = np.array(flux_values)
    
    if method == 'median':
        normalized = flux_array / np.median(flux_array)
    elif method == 'mean':
        normalized = flux_array / np.mean(flux_array)
    elif method == 'minmax':
        min_val = np.min(flux_array)
        max_val = np.max(flux_array)
        normalized = (flux_array - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    return normalized.tolist()
