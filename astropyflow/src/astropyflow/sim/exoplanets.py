"""
Simulação e classificação de exoplanetas.

Implementa:
- Classificação básica por raio, massa e período
- Cálculo de zona habitável
- Avaliação de habitabilidade potencial
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u

logger = logging.getLogger(__name__)


class PlanetClass(Enum):
    """Classes de planetas baseadas em raio."""
    TERRESTRIAL = "Terrestre"
    SUPER_EARTH = "Super-Terra"
    MINI_NEPTUNE = "Mini-Netuno"
    NEPTUNE_LIKE = "Netuniano"
    GAS_GIANT = "Gigante Gasoso"


class HabitabilityClass(Enum):
    """Classes de habitabilidade."""
    TOO_HOT = "Muito Quente"
    HOT = "Quente"
    HABITABLE = "Habitável"
    COLD = "Frio"
    TOO_COLD = "Muito Frio"


@dataclass
class ExoplanetData:
    """
    Dados de um exoplaneta.

    Attributes:
        name: Nome do exoplaneta.
        radius: Raio em raios terrestres (R_earth).
        mass: Massa em massas terrestres (M_earth).
        period: Período orbital em dias.
        semi_major_axis: Semi-eixo maior em AU.
        equilibrium_temp: Temperatura de equilíbrio em Kelvin.
        host_star_temp: Temperatura da estrela hospedeira em K.
        host_star_radius: Raio da estrela em raios solares.
        eccentricity: Excentricidade orbital.
    """
    name: str
    radius: Optional[float] = None
    mass: Optional[float] = None
    period: Optional[float] = None
    semi_major_axis: Optional[float] = None
    equilibrium_temp: Optional[float] = None
    host_star_temp: Optional[float] = None
    host_star_radius: Optional[float] = None
    eccentricity: float = 0.0


def classify_exoplanet(
    radius: Optional[float] = None,
    mass: Optional[float] = None,
    period: Optional[float] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Classifica exoplaneta baseado em propriedades físicas.

    Critérios de classificação (baseado em Rogers 2015, Chen & Kipping 2017):
    - Terrestre: R < 1.25 R_earth
    - Super-Terra: 1.25 <= R < 2.0 R_earth
    - Mini-Netuno: 2.0 <= R < 4.0 R_earth
    - Netuniano: 4.0 <= R < 8.0 R_earth
    - Gigante Gasoso: R >= 8.0 R_earth

    Args:
        radius: Raio em raios terrestres.
        mass: Massa em massas terrestres.
        period: Período orbital em dias.
        temperature: Temperatura de equilíbrio em K.

    Returns:
        Dicionário com:
            - class: Classe principal do planeta
            - subclass: Sub-classificação se aplicável
            - is_terrestrial: Booleano indicando se é rochoso
            - confidence: Confiança da classificação (0-1)
            - notes: Notas explicativas

    Example:
        >>> result = classify_exoplanet(radius=1.5, temperature=288)
        >>> print(result['class'].value)
        'Super-Terra'
    """
    result = {
        "class": None,
        "subclass": None,
        "is_terrestrial": False,
        "confidence": 0.0,
        "notes": [],
    }

    # Classificação por raio (mais confiável)
    if radius is not None:
        if radius < 1.25:
            result["class"] = PlanetClass.TERRESTRIAL
            result["is_terrestrial"] = True
            result["confidence"] = 0.9
            result["notes"].append("Planeta provavelmente rochoso")
        elif radius < 2.0:
            result["class"] = PlanetClass.SUPER_EARTH
            result["is_terrestrial"] = True
            result["confidence"] = 0.75
            result["notes"].append("Possivelmente rochoso com atmosfera espessa")
        elif radius < 4.0:
            result["class"] = PlanetClass.MINI_NEPTUNE
            result["confidence"] = 0.8
            result["notes"].append("Provavelmente possui envelope gasoso significativo")
        elif radius < 8.0:
            result["class"] = PlanetClass.NEPTUNE_LIKE
            result["confidence"] = 0.85
            result["notes"].append("Gigante de gelo, similar a Netuno")
        else:
            result["class"] = PlanetClass.GAS_GIANT
            result["confidence"] = 0.9
            result["notes"].append("Gigante gasoso, similar a Júpiter")

    # Classificação alternativa por massa (se raio não disponível)
    if result["class"] is None and mass is not None:
        if mass < 2.0:
            result["class"] = PlanetClass.TERRESTRIAL
            result["is_terrestrial"] = True
            result["confidence"] = 0.7
        elif mass < 10.0:
            result["class"] = PlanetClass.SUPER_EARTH
            result["is_terrestrial"] = True
            result["confidence"] = 0.6
        elif mass < 50.0:
            result["class"] = PlanetClass.MINI_NEPTUNE
            result["confidence"] = 0.65
        elif mass < 300.0:
            result["class"] = PlanetClass.NEPTUNE_LIKE
            result["confidence"] = 0.7
        else:
            result["class"] = PlanetClass.GAS_GIANT
            result["confidence"] = 0.8

    # Adicionar informação sobre temperatura
    if temperature is not None:
        if temperature > 1000:
            result["subclass"] = "Ultra-Hot"
            result["notes"].append("Temperatura extremamente alta")
        elif temperature > 500:
            result["subclass"] = "Hot"
            result["notes"].append("Planeta quente")
        elif temperature > 200:
            result["subclass"] = "Temperate"
            result["notes"].append("Temperatura moderada")
        else:
            result["subclass"] = "Cold"
            result["notes"].append("Planeta frio")

    if result["class"] is None:
        result["notes"].append("Dados insuficientes para classificação")
        result["confidence"] = 0.0

    return result


def calculate_habitable_zone(
    star_temp: float,
    star_radius: float,
    star_luminosity: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calcula limites da zona habitável para uma estrela.

    Usa modelo de Kopparapu et al. (2013) simplificado.

    Args:
        star_temp: Temperatura efetiva da estrela em K.
        star_radius: Raio da estrela em raios solares.
        star_luminosity: Luminosidade em luminosidades solares.
                        Se None, calcula a partir de T e R.

    Returns:
        Dicionário com limites em AU:
            - inner_recent_venus: Limite interior (Vênus recente)
            - inner_runaway: Efeito estufa descontrolado
            - outer_max_greenhouse: Limite exterior máximo
            - outer_early_mars: Limite exterior (Marte primitivo)
            - optimistic_inner: Limite otimista interior
            - optimistic_outer: Limite otimista exterior

    Notes:
        - Zona habitável conservadora: entre runaway greenhouse e max greenhouse
        - Zona habitável otimista: entre recent Venus e early Mars
    """
    # Calcular luminosidade se não fornecida
    # L/L_sun = (R/R_sun)^2 * (T/T_sun)^4
    t_sun = 5778  # K
    if star_luminosity is None:
        luminosity = (star_radius ** 2) * (star_temp / t_sun) ** 4
    else:
        luminosity = star_luminosity

    # Coeficientes simplificados de Kopparapu et al. (2013)
    # Para estrelas tipo G (como o Sol)
    s_eff_recent_venus = 1.776
    s_eff_runaway = 1.107
    s_eff_max_greenhouse = 0.356
    s_eff_early_mars = 0.320

    # Ajuste fino baseado na temperatura estelar (simplificado)
    temp_factor = (star_temp - 5778) / 5778
    s_eff_runaway *= (1 + 0.1 * temp_factor)
    s_eff_max_greenhouse *= (1 + 0.05 * temp_factor)

    # Calcular distâncias: d = sqrt(L / S_eff)
    habitable_zone = {
        "inner_recent_venus": np.sqrt(luminosity / s_eff_recent_venus),
        "inner_runaway": np.sqrt(luminosity / s_eff_runaway),
        "outer_max_greenhouse": np.sqrt(luminosity / s_eff_max_greenhouse),
        "outer_early_mars": np.sqrt(luminosity / s_eff_early_mars),
        "optimistic_inner": np.sqrt(luminosity / s_eff_recent_venus),
        "optimistic_outer": np.sqrt(luminosity / s_eff_early_mars),
        "luminosity_solar": luminosity,
    }

    logger.debug(f"Zona habitável calculada: {habitable_zone['inner_runaway']:.2f} - "
                 f"{habitable_zone['outer_max_greenhouse']:.2f} AU")

    return habitable_zone


def is_in_habitable_zone(
    semi_major_axis: float,
    star_temp: float,
    star_radius: float,
    eccentricity: float = 0.0,
    conservative: bool = True,
) -> Dict[str, Any]:
    """
    Verifica se planeta está na zona habitável.

    Args:
        semi_major_axis: Semi-eixo maior da órbita em AU.
        star_temp: Temperatura da estrela em K.
        star_radius: Raio da estrela em raios solares.
        eccentricity: Excentricidade orbital.
        conservative: Se True, usa limites conservadores.

    Returns:
        Dicionário com:
            - in_zone: Booleano indicando se está na zona
            - zone_class: Classe de habitabilidade
            - distance_from_center: Distância do centro da zona (AU)
            - flux_received: Fluxo recebido relativo à Terra
    """
    hz = calculate_habitable_zone(star_temp, star_radius)

    if conservative:
        inner_edge = hz["inner_runaway"]
        outer_edge = hz["outer_max_greenhouse"]
    else:
        inner_edge = hz["optimistic_inner"]
        outer_edge = hz["optimistic_outer"]

    # Centro da zona habitável
    hz_center = np.sqrt(inner_edge * outer_edge)
    hz_width = outer_edge - inner_edge

    # Considerar excentricidade: distância média varia
    # Para órbitas excêntricas, usar fluxo médio
    if eccentricity > 0:
        # Fluxo médio considerando órbita elíptica
        flux_avg = 1 / (semi_major_axis ** 2 * np.sqrt(1 - eccentricity ** 2))
        # Semi-eixo equivalente para fluxo médio
        effective_distance = np.sqrt(1 / flux_avg)
    else:
        effective_distance = semi_major_axis

    # Fluxo recebido relativo à Terra
    flux_received = luminosity_to_flux(semi_major_axis, star_temp, star_radius)

    # Determinar classe de habitabilidade
    if effective_distance < inner_edge:
        in_zone = False
        if effective_distance < inner_edge * 0.5:
            zone_class = HabitabilityClass.TOO_HOT
        else:
            zone_class = HabitabilityClass.HOT
    elif effective_distance > outer_edge:
        in_zone = False
        if effective_distance > outer_edge * 2:
            zone_class = HabitabilityClass.TOO_COLD
        else:
            zone_class = HabitabilityClass.COLD
    else:
        in_zone = True
        zone_class = HabitabilityClass.HABITABLE

    # Distância normalizada do centro da zona
    if hz_width > 0:
        distance_from_center = (effective_distance - hz_center) / hz_width
    else:
        distance_from_center = 0.0

    result = {
        "in_zone": in_zone,
        "zone_class": zone_class.value,
        "distance_from_center": distance_from_center,
        "flux_received": flux_received,
        "hz_bounds": {
            "inner": inner_edge,
            "outer": outer_edge,
            "center": hz_center,
        },
    }

    logger.info(f"Habitabilidade: {zone_class.value} (a={semi_major_axis:.2f} AU)")
    return result


def luminosity_to_flux(
    distance_au: float,
    star_temp: float,
    star_radius: float,
) -> float:
    """
    Calcula fluxo recebido por planeta relativo à Terra.

    Args:
        distance_au: Distância estrela-planeta em AU.
        star_temp: Temperatura da estrela em K.
        star_radius: Raio da estrela em raios solares.

    Returns:
        Fluxo em unidades do fluxo solar na Terra.
    """
    t_sun = 5778  # K
    luminosity = (star_radius ** 2) * (star_temp / t_sun) ** 4
    flux = luminosity / (distance_au ** 2)
    return flux


def calculate_equilibrium_temperature(
    semi_major_axis: float,
    star_temp: float,
    star_radius: float,
    albedo: float = 0.3,
    greenhouse_factor: float = 1.0,
) -> float:
    """
    Calcula temperatura de equilíbrio do planeta.

    Fórmula: T_eq = T_star * sqrt(R_star / 2a) * (1 - A)^(1/4) * G

    Args:
        semi_major_axis: Distância em AU.
        star_temp: Temperatura da estrela em K.
        star_radius: Raio da estrela em raios solares.
        albedo: Albedo de Bond (0-1). Default 0.3 (similar à Terra).
        greenhouse_factor: Fator de efeito estufa. Default 1.0 (sem efeito).

    Returns:
        Temperatura de equilíbrio em Kelvin.
    """
    r_sun_au = 0.00465  # Raio solar em AU

    # T_eq = T_star * sqrt(R_star / 2a) * (1 - A)^(1/4)
    t_eq = (
        star_temp
        * np.sqrt((star_radius * r_sun_au) / (2 * semi_major_axis))
        * (1 - albedo) ** 0.25
        * greenhouse_factor
    )

    return t_eq


def analyze_exoplanet_system(
    planets: List[ExoplanetData],
    star_temp: float = 5778,
    star_radius: float = 1.0,
) -> pd.DataFrame:
    """
    Analisa sistema de múltiplos exoplanetas.

    Args:
        planets: Lista de objetos ExoplanetData.
        star_temp: Temperatura da estrela em K.
        star_radius: Raio da estrela em raios solares.

    Returns:
        DataFrame com análise completa de cada planeta.
    """
    results = []

    for planet in planets:
        # Classificação
        classification = classify_exoplanet(
            radius=planet.radius,
            mass=planet.mass,
            temperature=planet.equilibrium_temp,
        )

        # Habitabilidade
        if planet.semi_major_axis is not None:
            habitability = is_in_habitable_zone(
                planet.semi_major_axis,
                star_temp,
                star_radius,
                planet.eccentricity,
            )
        else:
            habitability = {"in_zone": False, "zone_class": "Desconhecido"}

        # Calcular temperatura se não fornecida
        if planet.equilibrium_temp is None and planet.semi_major_axis is not None:
            eq_temp = calculate_equilibrium_temperature(
                planet.semi_major_axis,
                star_temp,
                star_radius,
            )
        else:
            eq_temp = planet.equilibrium_temp

        results.append({
            "name": planet.name,
            "radius_earth": planet.radius,
            "mass_earth": planet.mass,
            "period_days": planet.period,
            "semi_major_au": planet.semi_major_axis,
            "temp_k": eq_temp,
            "class": classification["class"].value if classification["class"] else "N/A",
            "is_terrestrial": classification["is_terrestrial"],
            "in_habitable_zone": habitability["in_zone"],
            "habitability_class": habitability["zone_class"],
            "confidence": classification["confidence"],
        })

    df = pd.DataFrame(results)
    logger.info(f"Sistema analisado: {len(df)} planetas")
    return df


def generate_mock_exoplanets(
    n_planets: int = 10,
    seed: Optional[int] = None,
) -> List[ExoplanetData]:
    """
    Gera dados mock de exoplanetas para teste.

    Args:
        n_planets: Número de planetas para gerar.
        seed: Seed para reprodutibilidade.

    Returns:
        Lista de ExoplanetData.
    """
    if seed is not None:
        np.random.seed(seed)

    planets = []

    for i in range(n_planets):
        # Gerar parâmetros realistas
        radius = np.random.lognormal(mean=0, sigma=0.5) * 1.5  # R_earth
        mass = radius ** 2.5 * np.random.uniform(0.8, 1.2)  # Relação massa-raio aproximada
        semi_major = np.random.lognormal(mean=0, sigma=0.8)  # AU
        period = np.sqrt(semi_major ** 3) * 365.25  # Lei de Kepler (anos -> dias)
        temp = calculate_equilibrium_temperature(semi_major, 5778, 1.0, albedo=0.3)

        planet = ExoplanetData(
            name=f"Planet_{chr(65 + i)}",  # Planeta_A, Planeta_B, ...
            radius=float(radius),
            mass=float(mass),
            period=float(period),
            semi_major_axis=float(semi_major),
            equilibrium_temp=float(temp),
            host_star_temp=5778,
            host_star_radius=1.0,
            eccentricity=np.random.uniform(0, 0.3),
        )
        planets.append(planet)

    logger.info(f"Gerados {n_planets} exoplanetas mock")
    return planets
