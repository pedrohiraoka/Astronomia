"""
Módulo de simulações do AstroPyFlow.

Contém simulações de:
- solar_system: Sistema solar simplificado (N-corpos 2D)
- exoplanets: Classificação e habitabilidade de exoplanetas
"""

from astropyflow.sim.solar_system import (
    simulate_orbits,
    create_body,
    integrate_verlet,
)
from astropyflow.sim.exoplanets import (
    classify_exoplanet,
    calculate_habitable_zone,
    is_in_habitable_zone,
)

__all__ = [
    "simulate_orbits",
    "create_body",
    "integrate_verlet",
    "classify_exoplanet",
    "calculate_habitable_zone",
    "is_in_habitable_zone",
]
