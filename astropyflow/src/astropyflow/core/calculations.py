"""
Cálculos astronômicos fundamentais.

Implementa funções para:
- Distância angular entre corpos celestes
- Distância linear a partir de paralaxe/coordenadas
- Luminosidade aparente e absoluta
- Conversões de unidades astronômicas
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord

logger = logging.getLogger(__name__)


def calculate_angular_distance(
    ra1: Union[float, npt.ArrayLike],
    dec1: Union[float, npt.ArrayLike],
    ra2: Union[float, npt.ArrayLike],
    dec2: Union[float, npt.ArrayLike],
    unit: u.UnitBase = u.deg,
) -> Union[u.Quantity, npt.NDArray]:
    """
    Calcula distância angular entre dois pontos na esfera celeste.

    Usa a fórmula do haversine para precisão em pequenas distâncias angulares.

    Args:
        ra1: Ascensão reta do primeiro ponto (em graus ou array).
        dec1: Declinação do primeiro ponto (em graus ou array).
        ra2: Ascensão reta do segundo ponto (em graus ou array).
        dec2: Declinação do segundo ponto (em graus ou array).
        unit: Unidade de retorno (default: graus).

    Returns:
        Distância angular na unidade especificada.

    Notes:
        - Fórmula: d = 2 * arcsin(sqrt(sin²(Δδ/2) + cos(δ1)*cos(δ2)*sin²(Δα/2)))
        - Funciona com valores escalares ou arrays NumPy (vetorizado).

    Example:
        >>> dist = calculate_angular_distance(0, 0, 1, 0)
        >>> print(f"Distância: {dist:.2f}")
        1.0 deg
    """
    # Converter para radianos para cálculos trigonométricos
    ra1_rad = np.radians(ra1) if isinstance(ra1, (int, float)) else np.radians(np.asarray(ra1))
    dec1_rad = np.radians(dec1) if isinstance(dec1, (int, float)) else np.radians(np.asarray(dec1))
    ra2_rad = np.radians(ra2) if isinstance(ra2, (int, float)) else np.radians(np.asarray(ra2))
    dec2_rad = np.radians(dec2) if isinstance(dec2, (int, float)) else np.radians(np.asarray(dec2))

    # Diferenças
    delta_ra = ra2_rad - ra1_rad
    delta_dec = dec2_rad - dec1_rad

    # Fórmula do haversine
    a = (
        np.sin(delta_dec / 2) ** 2
        + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(delta_ra / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip para evitar erros numéricos

    # Converter de radianos para unidade desejada
    distance_rad = c * u.rad
    distance = distance_rad.to(unit)

    logger.debug(f"Distância angular calculada: {distance}")
    return distance


def calculate_linear_distance(
    ra1: float,
    dec1: float,
    dist1: u.Quantity,
    ra2: float,
    dec2: float,
    dist2: u.Quantity,
) -> u.Quantity:
    """
    Calcula distância linear (3D) entre dois corpos celestes.

    Considera as distâncias individuais de cada corpo à Terra.

    Args:
        ra1: Ascensão reta do primeiro corpo (graus).
        dec1: Declinação do primeiro corpo (graus).
        dist1: Distância do primeiro corpo à Terra (Quantity Astropy).
        ra2: Ascensão reta do segundo corpo (graus).
        dec2: Declinação do segundo corpo (graus).
        dist2: Distância do segundo corpo à Terra (Quantity Astropy).

    Returns:
        Distância linear como Quantity Astropy na mesma unidade de entrada.

    Notes:
        - Converte coordenadas esféricas para cartesianas
        - Calcula distância euclidiana 3D
        - Fórmula: d = sqrt(r1² + r2² - 2*r1*r2*cos(θ)), onde θ é separação angular

    Example:
        >>> from astropy import units as u
        >>> dist = calculate_linear_distance(0, 0, 10*u.pc, 1, 0, 15*u.pc)
        >>> print(dist.to(u.ly))
    """
    # Calcular separação angular
    angular_sep = calculate_angular_distance(ra1, dec1, ra2, dec2, unit=u.rad)
    theta = angular_sep.value

    # Lei dos cossenos para triângulo 3D
    r1 = dist1
    r2 = dist2

    # d² = r1² + r2² - 2*r1*r2*cos(θ)
    cos_theta = np.cos(theta)
    d_squared = r1**2 + r2**2 - 2 * r1 * r2 * cos_theta

    # Evitar problemas numéricos com valores negativos muito pequenos
    d_squared = np.maximum(d_squared, 0 * d_squared.unit**2)

    linear_dist = np.sqrt(d_squared)

    logger.debug(f"Distância linear calculada: {linear_dist}")
    return linear_dist


def calculate_absolute_magnitude(
    apparent_mag: Union[float, npt.ArrayLike],
    distance: Union[u.Quantity, float],
    distance_unit: u.UnitBase = u.pc,
) -> Union[float, npt.NDArray]:
    """
    Calcula magnitude absoluta a partir da magnitude aparente e distância.

    Usa a fórmula do módulo de distância: M = m - 5*log10(d/10)

    Args:
        apparent_mag: Magnitude aparente (m).
        distance: Distância ao objeto. Se float, assume distance_unit.
        distance_unit: Unidade da distância (default: parsecs).

    Returns:
        Magnitude absoluta (M).

    Notes:
        -Magnitude absoluta é definida como magnitude a 10 pc de distância
        - Fórmula: M = m - 5*log10(d) + 5 = m - 5*log10(d/10)

    Example:
        >>> M = calculate_absolute_magnitude(5.0, 100)  # 100 pc
        >>> print(f"Magnitude absoluta: {M:.2f}")
    """
    if isinstance(distance, u.Quantity):
        dist_pc = distance.to(u.pc).value
    else:
        dist_pc = distance

    # Evitar log de zero ou negativo
    dist_pc = np.maximum(dist_pc, 1e-10)

    # Módulo de distância: m - M = 5*log10(d) - 5
    absolute_mag = np.asarray(apparent_mag) - 5 * np.log10(dist_pc / 10)

    logger.debug(f"Magnitude absoluta calculada: {absolute_mag}")
    return absolute_mag


def calculate_luminosity(
    apparent_mag: Union[float, npt.ArrayLike],
    distance: Union[u.Quantity, float],
    bolometric_correction: float = 0.0,
    reference_mag: float = 4.74,  # M_bol do Sol
) -> Union[float, npt.NDArray]:
    """
    Calcula luminosidade em unidades solares.

    Args:
        apparent_mag: Magnitude aparente do objeto.
        distance: Distância ao objeto (preferencialmente em Quantity).
        bolometric_correction: Correção bolométrica (default: 0).
        reference_mag: Magnitude bolométrica de referência (default: 4.74 para o Sol).

    Returns:
        Luminosidade em unidades solares (L/L_sun).

    Notes:
        - Primeiro calcula magnitude absoluta
        - Aplica correção bolométrica
        - Converte para luminosidade: L/L_sun = 10^((M_sun - M)/2.5)

    Example:
        >>> L = calculate_luminosity(0.0, 10*u.pc)
        >>> print(f"Luminosidade: {L:.2f} L_sun")
    """
    # Calcular magnitude absoluta
    abs_mag = calculate_absolute_magnitude(apparent_mag, distance)

    # Aplicar correção bolométrica
    m_bol = abs_mag - bolometric_correction

    # Converter para luminosidade
    # L/L_sun = 10^((M_bol,sun - M_bol)/2.5)
    luminosity_ratio = 10 ** ((reference_mag - m_bol) / 2.5)

    logger.debug(f"Luminosidade calculada: {luminosity_ratio} L_sun")
    return luminosity_ratio


def convert_units(
    value: Union[float, npt.ArrayLike, u.Quantity],
    from_unit: Union[str, u.UnitBase],
    to_unit: Union[str, u.UnitBase],
) -> Union[float, npt.NDArray, u.Quantity]:
    """
    Converte valores entre unidades astronômicas.

    Args:
        value: Valor ou array para converter. Pode ser Quantity.
        from_unit: Unidade de origem (string ou Unit).
        to_unit: Unidade de destino (string ou Unit).

    Returns:
        Valor convertido. Retorna Quantity se input for Quantity, caso contrário float/array.

    Raises:
        u.UnitsError: Se conversão não for possível.

    Example:
        >>> convert_units(100, 'pc', 'ly')
        >>> convert_units(5.0 * u.mag, u.mag, u.dimensionless_unscaled)
    """
    # Converter strings para unidades Astropy
    if isinstance(from_unit, str):
        from_unit = u.Unit(from_unit)
    if isinstance(to_unit, str):
        to_unit = u.Unit(to_unit)

    # Se value já for Quantity, usar sua conversão nativa
    if isinstance(value, u.Quantity):
        return value.to(to_unit)

    # Caso contrário, criar Quantity temporário
    quantity = np.asarray(value) * from_unit
    result = quantity.to(to_unit)

    # Se resultado for adimensional, retornar apenas valor numérico
    if to_unit == u.dimensionless_unscaled:
        return result.value

    logger.debug(f"Conversão: {value} {from_unit} -> {result}")
    return result


def parsec_to_lightyear(distance_pc: Union[float, npt.ArrayLike]) -> Union[float, npt.NDArray]:
    """
    Converte distância de parsecs para anos-luz.

    Args:
        distance_pc: Distância em parsecs.

    Returns:
        Distância em anos-luz.

    Example:
        >>> ly = parsec_to_lightyear(10)
        >>> print(f"{ly:.2f} anos-luz")
    """
    return convert_units(distance_pc, u.pc, u.lightyear)


def lightyear_to_parsec(distance_ly: Union[float, npt.ArrayLike]) -> Union[float, npt.NDArray]:
    """
    Converte distância de anos-luz para parsecs.

    Args:
        distance_ly: Distância em anos-luz.

    Returns:
        Distância em parsecs.
    """
    return convert_units(distance_ly, u.lightyear, u.pc)


def magnitude_to_flux(
    magnitude: Union[float, npt.ArrayLike],
    zero_point: float = 3631,  # Jy para sistema AB
    band: str = "V",
) -> Union[float, npt.NDArray]:
    """
    Converte magnitude para fluxo físico.

    Args:
        magnitude: Magnitude aparente.
        zero_point: Fluxo de referência em Janskys (default: 3631 Jy para AB).
        band: Banda fotométrica (V, B, R, etc.) - usado para selecionar zero_point.

    Returns:
        Fluxo em Janskys.

    Notes:
        - Fórmula: F = F0 * 10^(-m/2.5)
        - Zero points comuns: V=3631, B=4260, K=670 Jy

    Example:
        >>> flux = magnitude_to_flux(0.0)  # Vega ~ 0 mag
        >>> print(f"Fluxo: {flux:.2f} Jy")
    """
    # Zero points aproximados por banda (Jy)
    zero_points = {
        "U": 1810,
        "B": 4260,
        "V": 3631,
        "R": 3080,
        "I": 2550,
        "J": 1600,
        "H": 1020,
        "K": 670,
    }

    zp = zero_points.get(band.upper(), zero_point)

    flux = zp * 10 ** (-np.asarray(magnitude) / 2.5)

    logger.debug(f"Fluxo calculado: {flux} Jy (banda {band})")
    return flux


def flux_to_magnitude(
    flux: Union[float, npt.ArrayLike],
    zero_point: float = 3631,
    band: str = "V",
) -> Union[float, npt.NDArray]:
    """
    Converte fluxo físico para magnitude.

    Args:
        flux: Fluxo em Janskys.
        zero_point: Fluxo de referência em Janskys.
        band: Banda fotométrica.

    Returns:
        Magnitude aparente.

    Notes:
        - Fórmula inversa: m = -2.5 * log10(F/F0)

    Example:
        >>> mag = flux_to_magnitude(3631)  # Should be ~0
        >>> print(f"Magnitude: {mag:.2f}")
    """
    flux_arr = np.asarray(flux)

    # Evitar log de zero
    flux_arr = np.maximum(flux_arr, 1e-30)

    magnitude = -2.5 * np.log10(flux_arr / zero_point)

    logger.debug(f"Magnitude calculada: {magnitude} (banda {band})")
    return magnitude
