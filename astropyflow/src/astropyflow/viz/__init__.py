"""
Módulo de visualização do AstroPyFlow.

Contém funções para plotagem de:
- Posição do Sol (altitude vs azimute)
- Curvas de luz de exoplanetas
- Outros gráficos astronômicos com Matplotlib + Astropy
"""

from astropyflow.viz.plotting import (
    plot_sun_position,
    plot_light_curve,
    plot_sky_distribution,
    save_plot,
)

__all__ = [
    "plot_sun_position",
    "plot_light_curve",
    "plot_sky_distribution",
    "save_plot",
]
