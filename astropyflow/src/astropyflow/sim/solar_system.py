"""
Simulação de sistema solar simplificada.

Implementa integração numérica de órbitas 2D usando:
- Método de Verlet (velocity-verlet) para estabilidade energética
- Lei da gravitação universal de Newton
- Vetorização NumPy para múltiplos corpos
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class CelestialBody:
    """
    Representa um corpo celeste em simulação orbital.

    Attributes:
        name: Nome do corpo.
        mass: Massa em kg.
        position: Posição inicial (x, y) em metros.
        velocity: Velocidade inicial (vx, vy) em m/s.
        color: Cor para visualização.
        radius: Raio físico em metros (para escala visual).
    """
    name: str
    mass: float
    position: npt.NDArray = field(default_factory=lambda: np.zeros(2))
    velocity: npt.NDArray = field(default_factory=lambda: np.zeros(2))
    color: str = "gray"
    radius: float = 1e6  # metros

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)


def create_body(
    name: str,
    mass: float,
    semi_major_axis: float,
    eccentricity: float = 0.0,
    initial_angle: float = 0.0,
    central_mass: float = 1.989e30,  # Massa do Sol
    color: str = "gray",
) -> CelestialBody:
    """
    Cria um corpo celeste com parâmetros orbitais keplerianos.

    Args:
        name: Nome do corpo.
        mass: Massa do corpo em kg.
        semi_major_axis: Semi-eixo maior da órbita em metros.
        eccentricity: Excentricidade orbital (0 = circular).
        initial_angle: Ângulo inicial em radianos.
        central_mass: Massa do corpo central (estrela) em kg.
        color: Cor para visualização.

    Returns:
        CelestialBody configurado com posição e velocidade iniciais.

    Notes:
        - Calcula velocidade orbital usando v = sqrt(GM/a) para órbita circular
        - Para órbitas elípticas, ajusta velocidade no periastro
    """
    G = 6.67430e-11  # Constante gravitacional m³/(kg·s²)

    # Posição inicial (coordenadas polares -> cartesianas)
    # Para órbita elíptica, começar no periastro: r = a(1-e)
    r = semi_major_axis * (1 - eccentricity)
    x = r * np.cos(initial_angle)
    y = r * np.sin(initial_angle)

    # Velocidade orbital
    # Para órbita circular: v = sqrt(GM/a)
    # Para elíptica no periastro: v = sqrt(GM(1+e)/(a(1-e)))
    if eccentricity < 1.0:
        v = np.sqrt(G * central_mass * (1 + eccentricity) / (semi_major_axis * (1 - eccentricity)))
    else:
        v = np.sqrt(G * central_mass / semi_major_axis)

    # Velocidade perpendicular ao vetor posição (órbita no sentido anti-horário)
    vx = -v * np.sin(initial_angle)
    vy = v * np.cos(initial_angle)

    return CelestialBody(
        name=name,
        mass=mass,
        position=np.array([x, y]),
        velocity=np.array([vx, vy]),
        color=color,
    )


def compute_accelerations(
    bodies: List[CelestialBody],
    G: float = 6.67430e-11,
) -> List[npt.NDArray]:
    """
    Calcula acelerações gravitacionais para todos os corpos.

    Usa lei de Newton: F = G*m1*m2/r², vetorializado.

    Args:
        bodies: Lista de corpos celestes.
        G: Constante gravitacional.

    Returns:
        Lista de arrays de aceleração (ax, ay) para cada corpo.
    """
    n = len(bodies)
    accelerations = [np.zeros(2) for _ in range(n)]

    # Calcular acelerações pairwise (O(n²), mas vetorizado internamente)
    for i in range(n):
        for j in range(i + 1, n):
            # Vetor de i para j
            r_vec = bodies[j].position - bodies[i].position
            r_mag = np.linalg.norm(r_vec)

            # Evitar singularidade (colisões)
            if r_mag < 1e6:  # Menor que 1000 km
                continue

            # Aceleração: a_i = G * m_j * r_hat / r²
            r_hat = r_vec / r_mag
            accel_mag = G / (r_mag ** 2)

            accelerations[i] += accel_mag * bodies[j].mass * r_hat
            accelerations[j] -= accel_mag * bodies[i].mass * r_hat  # Ação e reação

    return accelerations


def integrate_verlet(
    bodies: List[CelestialBody],
    dt: float,
    steps: int,
    G: float = 6.67430e-11,
) -> Dict[str, npt.NDArray]:
    """
    Integra equações de movimento usando Velocity Verlet.

    Método symplectic que conserva energia melhor que Euler.

    Algoritmo:
        1. v(t+dt/2) = v(t) + a(t)*dt/2
        2. r(t+dt) = r(t) + v(t+dt/2)*dt
        3. a(t+dt) = F(r(t+dt))/m
        4. v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2

    Args:
        bodies: Lista de corpos celestes.
        dt: Passo de tempo em segundos.
        steps: Número de passos de integração.
        G: Constante gravitacional.

    Returns:
        Dicionário com trajetórias: {nome_corpo: array(steps, 2)}
    """
    n_bodies = len(bodies)

    # Arrays para armazenar trajetórias
    trajectories = {b.name: np.zeros((steps, 2)) for b in bodies}

    # Calcular acelerações iniciais
    accelerations = compute_accelerations(bodies, G)

    logger.info(f"Iniciando integração Verlet: {steps} passos, dt={dt:.2e}s")

    for step in range(steps):
        # Salvar posições atuais
        for i, body in enumerate(bodies):
            trajectories[body.name][step] = body.position.copy()

        # Meio passo de velocidade: v(t+dt/2) = v(t) + a(t)*dt/2
        for i, body in enumerate(bodies):
            body.velocity += accelerations[i] * dt / 2

        # Passo completo de posição: r(t+dt) = r(t) + v(t+dt/2)*dt
        for i, body in enumerate(bodies):
            body.position += body.velocity * dt

        # Recalcular acelerações nas novas posições
        new_accelerations = compute_accelerations(bodies, G)

        # Segundo meio passo de velocidade: v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
        for i, body in enumerate(bodies):
            body.velocity += new_accelerations[i] * dt / 2

        # Atualizar acelerações
        accelerations = new_accelerations

        if step % (steps // 10) == 0 and step > 0:
            logger.debug(f"Integração: {step/steps*100:.0f}% completo")

    logger.info("Integração Verlet concluída")
    return trajectories


def simulate_orbits(
    n_bodies: int = 3,
    steps: int = 1000,
    dt_days: float = 1.0,
    total_time_years: Optional[float] = None,
    seed: Optional[int] = None,
    include_sun: bool = True,
) -> Dict[str, Any]:
    """
    Simula sistema de N-corpos com configuração simplificada.

    Args:
        n_bodies: Número total de corpos (inclui Sol se include_sun=True).
        steps: Número de passos de integração.
        dt_days: Passo de tempo em dias.
        total_time_years: Tempo total de simulação em anos (sobrescreve steps).
        seed: Seed para reprodutibilidade.
        include_sun: Incluir corpo central massivo (Sol).

    Returns:
        Dicionário com:
            - trajectories: Dict com trajetórias de cada corpo
            - bodies: Lista de corpos simulados
            - time_array: Array de tempos em dias
            - info: Informações da simulação

    Example:
        >>> result = simulate_orbits(n_bodies=4, steps=500, dt_days=0.5)
        >>> trajectories = result['trajectories']
    """
    if seed is not None:
        np.random.seed(seed)

    G = 6.67430e-11
    M_sun = 1.989e30  # kg
    AU = 1.496e11     # metros

    dt_seconds = dt_days * 24 * 3600

    if total_time_years is not None:
        steps = int(total_time_years * 365.25 / dt_days)

    bodies: List[CelestialBody] = []

    # Adicionar Sol
    if include_sun:
        sun = CelestialBody(
            name="Sun",
            mass=M_sun,
            position=np.zeros(2),
            velocity=np.zeros(2),
            color="#FFD700",
            radius=6.96e8,
        )
        bodies.append(sun)
        remaining_bodies = n_bodies - 1
    else:
        remaining_bodies = n_bodies

    # Configurações planetárias aproximadas (para sistema solar)
    planet_configs = [
        ("Mercury", 3.30e23, 0.387, 0.206, "#B5B5B5"),
        ("Venus", 4.87e24, 0.723, 0.007, "#E3BB76"),
        ("Earth", 5.97e24, 1.000, 0.017, "#4A90D9"),
        ("Mars", 6.42e23, 1.524, 0.094, "#D14A28"),
        ("Jupiter", 1.90e27, 5.203, 0.049, "#D4A574"),
        ("Saturn", 5.68e26, 9.537, 0.057, "#F4D59E"),
        ("Uranus", 8.68e25, 19.19, 0.046, "#BCE0F0"),
        ("Neptune", 1.02e26, 30.07, 0.010, "#5B7C99"),
    ]

    # Criar corpos
    for i in range(remaining_bodies):
        if i < len(planet_configs):
            name, mass, a_au, ecc, color = planet_configs[i]
            semi_major = a_au * AU
        else:
            # Gerar planeta aleatório
            name = f"Planet_{i+1}"
            mass = np.random.uniform(1e23, 1e26)
            semi_major = np.random.uniform(0.5, 10) * AU
            ecc = np.random.uniform(0, 0.3)
            color = np.random.choice(["#FF6B6B", "#4ECDC4", "#95E1D3", "#F9E79F"])

        angle = np.random.uniform(0, 2 * np.pi)
        body = create_body(
            name=name,
            mass=mass,
            semi_major_axis=semi_major,
            eccentricity=ecc,
            initial_angle=angle,
            central_mass=M_sun,
            color=color,
        )
        bodies.append(body)

    # Executar integração
    trajectories = integrate_verlet(bodies, dt_seconds, steps, G)

    # Array de tempos
    time_array = np.arange(steps) * dt_days

    info = {
        "n_bodies": len(bodies),
        "steps": steps,
        "dt_days": dt_days,
        "total_days": steps * dt_days,
        "total_years": steps * dt_days / 365.25,
        "G": G,
        "include_sun": include_sun,
    }

    logger.info(f"Simulação concluída: {info['total_years']:.2f} anos simulados")

    return {
        "trajectories": trajectories,
        "bodies": bodies,
        "time_array": time_array,
        "info": info,
    }


def plot_orbits(
    trajectories: Dict[str, npt.NDArray],
    bodies: List[CelestialBody],
    output_path: Optional[str] = None,
    scale_au: bool = True,
) -> Tuple[Any, Any]:
    """
    Plota órbitas resultantes da simulação.

    Args:
        trajectories: Dict de trajetórias da simulação.
        bodies: Lista de corpos celestes.
        output_path: Caminho para salvar o gráfico.
        scale_au: Se True, escala eixos em AU.

    Returns:
        Tupla (figura, eixo).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    AU = 1.496e11

    for body in bodies:
        if body.name in trajectories:
            traj = trajectories[body.name]
            x = traj[:, 0] / AU if scale_au else traj[:, 0]
            y = traj[:, 1] / AU if scale_au else traj[:, 1]

            ax.plot(x, y, color=body.color, linewidth=1, label=body.name, alpha=0.7)

            # Marcar posição inicial
            ax.scatter(x[0], y[0], color=body.color, s=50, marker='o')

    ax.set_xlabel("X (AU)" if scale_au else "X (m)")
    ax.set_ylabel("Y (AU)" if scale_au else "Y (m)")
    ax.set_title("Órbitas Simuladas")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax
