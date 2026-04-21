"""
Funções de plotagem para visualização astronômica.

Implementa gráficos com Matplotlib + Astropy:
- Posição do Sol (altitude vs azimute) ao longo do dia
- Curva de luz de exoplanetas (fluxo vs tempo)
- Distribuição de objetos no céu
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

logger = logging.getLogger(__name__)


def plot_sun_position(
    location: Union[str, EarthLocation] = "Sao Paulo",
    date: Union[str, datetime, Time] = None,
    timezone_offset: float = -3.0,
    output_path: Optional[Union[str, Path]] = None,
    show_grid: bool = True,
    **plot_kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plota a posição do Sol (Altitude vs Azimute) para uma localização e data.

    Args:
        location: Nome da cidade ou objeto EarthLocation.
                  Cidades suportadas: 'Sao Paulo', 'Rio de Janeiro', 'Lisboa', etc.
        date: Data para o cálculo. Se None, usa data atual.
        timezone_offset: Offset do fuso horário em horas (default: -3 para Brasil).
        output_path: Caminho para salvar o gráfico. Se None, não salva.
        show_grid: Mostra grid no plot.
        **plot_kwargs: Argumentos adicionais para matplotlib.

    Returns:
        Tupla (figura, eixo) do Matplotlib.

    Example:
        >>> fig, ax = plot_sun_position("Sao Paulo", "2024-01-15")
        >>> plt.show()
    """
    # Configurar localização
    if isinstance(location, str):
        location = get_earth_location(location)

    # Configurar data
    if date is None:
        date = datetime.now()
    elif isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    time_obj = Time(date)

    # Gerar array de tempos ao longo do dia (24 horas, passo de 5 minutos)
    times = time_obj + np.linspace(0, 1, 288) * u.day  # 288 intervalos de 5 min

    # Coordenadas do Sol - usar get_body para posição precisa
    from astropy.coordinates import get_body
    
    altitudes = []
    azimuths = []
    
    for t in times:
        sun_coord = get_body("sun", t, location=location)
        altaz_frame = AltAz(obstime=t, location=location)
        sun_altaz = sun_coord.transform_to(altaz_frame)
        altitudes.append(sun_altaz.alt.deg)
        azimuths.append(sun_altaz.az.deg)
    
    altitude = np.array(altitudes)
    azimuth = np.array(azimuths)

    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotar trajetória
    color = plot_kwargs.get("color", "#FF6B35")
    ax.plot(times.datetime, altitude, color=color, linewidth=2, label="Sol")

    # Marcar nascer e pôr do sol (altitude = 0)
    above_horizon = altitude > 0
    if np.any(above_horizon):
        idx = np.where(above_horizon)[0]
        sunrise_idx = idx[0]
        sunset_idx = idx[-1]

        ax.axvline(x=times[sunrise_idx].datetime, color="green", linestyle="--", alpha=0.7, label="Nascer")
        ax.axvline(x=times[sunset_idx].datetime, color="red", linestyle="--", alpha=0.7, label="Pôr")

        # Duração do dia
        day_duration = (times[sunset_idx] - times[sunrise_idx]).to(u.hour).value
        ax.set_title(f"Posição do Sol - {location.name}\n{date.strftime('%Y-%m-%d')} | Duração: {day_duration:.1f}h")
    else:
        ax.set_title(f"Posição do Sol - {location.name}\n{date.strftime('%Y-%m-%d')} | Sol não nasce neste dia")

    # Configurar eixos
    ax.set_xlabel("Hora Local (UTC{})".format(timezone_offset), fontsize=11)
    ax.set_ylabel("Altitude (graus)", fontsize=11)

    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="-")

    ax.legend(loc="best")
    ax.set_ylim(-90, 90)

    # Linha do horizonte
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5, label="Horizonte")

    plt.tight_layout()

    # Salvar se caminho especificado
    if output_path:
        save_plot(fig, output_path)

    logger.info(f"Gráfico de posição solar gerado para {location.name} em {date}")
    return fig, ax


def plot_light_curve(
    time_data: Union[npt.ArrayLike, List[float]],
    flux_data: Union[npt.ArrayLike, List[float]],
    flux_error: Optional[Union[npt.ArrayLike, List[float]]] = None,
    title: str = "Curva de Luz",
    xlabel: str = "Tempo (dias)",
    ylabel: str = "Fluxo Relativo",
    output_path: Optional[Union[str, Path]] = None,
    show_transit: bool = False,
    transit_center: Optional[float] = None,
    transit_depth: Optional[float] = None,
    **plot_kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plota curva de luz de exoplaneta (fluxo vs tempo).

    Args:
        time_data: Array de tempos (dias ou fase orbital).
        flux_data: Array de fluxos relativos.
        flux_error: Erros nos fluxos (opcional, para barras de erro).
        title: Título do gráfico.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        output_path: Caminho para salvar o gráfico.
        show_transit: Destacar região de trânsito.
        transit_center: Centro do trânsito em unidades de tempo.
        transit_depth: Profundidade do trânsito para destacar.
        **plot_kwargs: Argumentos adicionais para matplotlib.

    Returns:
        Tupla (figura, eixo) do Matplotlib.

    Example:
        >>> time = np.linspace(0, 10, 1000)
        >>> flux = 1 - 0.01 * np.exp(-((time - 5) ** 2) / 0.1)
        >>> fig, ax = plot_light_curve(time, flux, show_transit=True, transit_center=5)
    """
    time_arr = np.asarray(time_data)
    flux_arr = np.asarray(flux_data)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plotar dados
    marker = plot_kwargs.get("marker", ".")
    markersize = plot_kwargs.get("markersize", 3)
    color = plot_kwargs.get("color", "#1E88E5")

    if flux_error is not None:
        error_arr = np.asarray(flux_error)
        ax.errorbar(
            time_arr, flux_arr, yerr=error_arr,
            fmt=marker, markersize=markersize, color=color,
            alpha=0.6, capsize=2, label="Dados"
        )
    else:
        ax.plot(time_arr, flux_arr, marker=marker, markersize=markersize,
                linestyle="", color=color, alpha=0.6, label="Dados")

    # Destacar trânsito se solicitado
    if show_transit and transit_center is not None:
        # Estimar duração do trânsito (ou usar valor fixo)
        transit_duration = plot_kwargs.get("transit_duration", 0.2)

        ax.axvspan(
            transit_center - transit_duration/2,
            transit_center + transit_duration/2,
            alpha=0.2, color="orange", label="Trânsito"
        )

        if transit_depth is not None:
            ax.axhline(
                y=1 - transit_depth, color="red",
                linestyle="--", alpha=0.7, label=f"Profundidade: {transit_depth:.3f}"
            )

    # Configurar gráfico
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    ax.grid(True, alpha=0.3, linestyle="-")
    ax.legend(loc="best")

    # Linha de fluxo base
    ax.axhline(y=1.0, color="gray", linestyle="-", alpha=0.5, label="Linha base")

    plt.tight_layout()

    if output_path:
        save_plot(fig, output_path)

    logger.info(f"Curva de luz plotada: {len(time_arr)} pontos")
    return fig, ax


def plot_sky_distribution(
    ra: Union[npt.ArrayLike, List[float]],
    dec: Union[npt.ArrayLike, List[float]],
    magnitudes: Optional[Union[npt.ArrayLike, List[float]]] = None,
    title: str = "Distribuição no Céu",
    projection: str = "aitoff",
    output_path: Optional[Union[str, Path]] = None,
    **plot_kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plota distribuição de objetos celestes em projeção esférica.

    Args:
        ra: Ascensão reta em graus.
        dec: Declinação em graus.
        magnitudes: Magnitudes aparentes (para tamanho/cor dos pontos).
        title: Título do gráfico.
        projection: Projeção do mapa ('aitoff', 'mollweide', 'hammer').
        output_path: Caminho para salvar.
        **plot_kwargs: Argumentos adicionais.

    Returns:
        Tupla (figura, eixo) do Matplotlib.

    Example:
        >>> ra = np.random.uniform(0, 360, 500)
        >>> dec = np.random.uniform(-90, 90, 500)
        >>> fig, ax = plot_sky_distribution(ra, dec, projection='aitoff')
    """
    ra_arr = np.radians(np.asarray(ra))
    dec_arr = np.radians(np.asarray(dec))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection=projection)

    # Tamanho baseado na magnitude (se disponível)
    if magnitudes is not None:
        mag_arr = np.asarray(magnitudes)
        # Inverter: objetos mais brilhantes (menor mag) = maiores pontos
        sizes = 10 ** ((20 - np.clip(mag_arr, 0, 20)) / 5) * 2
        colors = mag_arr
        scatter = ax.scatter(ra_arr, dec_arr, s=sizes, c=colors,
                            cmap="viridis_r", alpha=0.6, edgecolors="none")
        plt.colorbar(scatter, label="Magnitude", pad=0.1)
    else:
        ax.scatter(ra_arr, dec_arr, s=5, alpha=0.5, color="#1E88E5")

    ax.set_title(title, pad=20)
    ax.grid(True, alpha=0.3)

    # Labels em radianos para projeções esféricas
    ax.set_xticklabels(["π", "3π/2", "0", "π/2", "π"])
    ax.set_yticklabels(["-π/2", "-π/4", "0", "π/4", "π/2"])

    plt.tight_layout()

    if output_path:
        save_plot(fig, output_path)

    logger.info(f"Distribuição celeste plotada: {len(ra_arr)} objetos")
    return fig, ax


def save_plot(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    formats: Optional[List[str]] = None,
) -> List[Path]:
    """
    Salva figura do Matplotlib em arquivo(s).

    Args:
        fig: Figura do Matplotlib.
        output_path: Caminho base (extensão será adicionada).
        dpi: Resolução em DPI (default: 300).
        formats: Lista de formatos ['png', 'svg', 'pdf']. Se None, detecta da extensão.

    Returns:
        Lista de caminhos dos arquivos salvos.

    Example:
        >>> paths = save_plot(fig, "output/plot.png")
        >>> paths = save_plot(fig, "output/plot", formats=["png", "svg"])
    """
    path = Path(output_path)
    saved_paths = []

    if formats is None:
        # Detectar formato da extensão
        if path.suffix:
            formats = [path.suffix[1:]]  # Remove o ponto
            base_path = path.with_suffix("")
        else:
            formats = ["png"]
            base_path = path
    else:
        base_path = path.with_suffix("")

    for fmt in formats:
        output_file = base_path.with_suffix(f".{fmt}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            fig.savefig(output_file, dpi=dpi, bbox_inches="tight", format=fmt)
            saved_paths.append(output_file)
            logger.info(f"Gráfico salvo: {output_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar {output_file}: {e}")
            raise

    return saved_paths


def get_earth_location(city_name: str) -> EarthLocation:
    """
    Retorna EarthLocation para cidades comuns.

    Args:
        city_name: Nome da cidade.

    Returns:
        EarthLocation com coordenadas da cidade.

    Raises:
        ValueError: Se cidade não for encontrada.
    """
    locations = {
        "sao paulo": (-23.5505 * u.deg, -46.6333 * u.deg, 760 * u.m),
        "rio de janeiro": (-22.9068 * u.deg, -43.1729 * u.deg, 11 * u.m),
        "lisboa": (38.7223 * u.deg, -9.1393 * u.deg, 100 * u.m),
        "porto": (41.1579 * u.deg, -8.6291 * u.deg, 100 * u.m),
        "madrid": (40.4168 * u.deg, -3.7038 * u.deg, 650 * u.m),
        "nova york": (40.7128 * u.deg, -74.0060 * u.deg, 10 * u.m),
        "tokyo": (35.6762 * u.deg, 139.6503 * u.deg, 40 * u.m),
        "londres": (51.5074 * u.deg, -0.1278 * u.deg, 11 * u.m),
        "paris": (48.8566 * u.deg, 2.3522 * u.deg, 35 * u.m),
        "berlim": (52.5200 * u.deg, 13.4050 * u.deg, 34 * u.m),
    }

    city_lower = city_name.lower()
    if city_lower not in locations:
        # Valores default para São Paulo se não encontrar
        logger.warning(f"Cidade '{city_name}' não encontrada. Usando São Paulo como default.")
        city_lower = "sao paulo"

    lat, lon, height = locations[city_lower]
    return EarthLocation(lat=lat, lon=lon, height=height)


def generate_mock_light_curve(
    n_points: int = 1000,
    period: float = 3.5,
    transit_depth: float = 0.01,
    transit_duration: float = 0.15,
    noise_level: float = 0.001,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera curva de luz simulada de exoplaneta em trânsito.

    Args:
        n_points: Número de pontos de dados.
        period: Período orbital em dias.
        transit_depth: Profundidade do trânsito (fração do fluxo).
        transit_duration: Duração do trânsito em dias.
        noise_level: Nível de ruído gaussiano.
        seed: Seed para reprodutibilidade.

    Returns:
        Tupla (tempo, fluxo, erro) como arrays NumPy.

    Example:
        >>> time, flux, error = generate_mock_light_curve(period=3.5, transit_depth=0.01)
        >>> fig, ax = plot_light_curve(time, flux)
    """
    if seed is not None:
        np.random.seed(seed)

    # Gerar tempos cobrindo múltiplos períodos
    time = np.linspace(0, period * 3, n_points)

    # Calcular fase orbital
    phase = (time % period) / period

    # Centro do trânsito na fase 0.5
    transit_center = 0.5
    half_duration = transit_duration / (2 * period)

    # Modelo simples de trânsito (caixa retangular suavizada)
    flux = np.ones(n_points)

    # Identificar pontos em trânsito
    in_transit = np.abs(phase - transit_center) < half_duration
    in_transit |= np.abs(phase - transit_center - 1) < half_duration  # Wrap around

    # Aplicar profundidade do trânsito com bordas suavizadas
    flux[in_transit] -= transit_depth

    # Adicionar ruído
    noise = np.random.normal(0, noise_level, n_points)
    flux += noise

    # Erros estimados
    error = np.ones(n_points) * noise_level

    return time, flux, error
