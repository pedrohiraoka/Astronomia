"""
Simulation of gravitational interaction between Sun, Earth, and Super Jupiter.
Uses Newton's Law of Universal Gravitation with RK4 numerical integration.
"""

from typing import Tuple
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.lines import Line2D
from tqdm.trange import trange

# Physical constants and normalization
G = 6.673e-11  # Gravitational constant (m^3 kg^-1 s^-2)
AU = 1.496e11  # Astronomical Unit (m)
YEAR = 365 * 24 * 3600 * 1.0  # One year in seconds
MM = 6e24  # Reference mass (kg) - Earth mass scale
ME = 1.0  # Earth mass in units of MM
MS = 2e30 / MM  # Sun mass in units of MM
MJ = 500 * 1.9e27 / MM  # Super Jupiter mass (500x Jupiter) in units of MM
GG = (MM * G * YEAR**2) / AU**3  # Normalized gravitational constant

# Simulation parameters
N = 120 * 100  # Total steps: 120 years * 100 steps/year
h = 1.0 / 100.0  # Time step in years


def gravitational_force(m1: float, m2: float, r: np.ndarray) -> np.ndarray:
    """
    Calculate gravitational force vector between two masses.
    
    Args:
        m1: Mass of first body (normalized units)
        m2: Mass of second body (normalized units)
        r: Position vector from m1 to m2 (in AU)
    
    Returns:
        Force vector acting on m2 due to m1 (normalized units)
    """
    epsilon = 1e-20
    norm_r = np.sqrt(np.sum(r**2) + epsilon)
    angle = np.arctan2(r[1], r[0] + epsilon)
    fx = np.cos(angle)
    fy = np.sin(angle)
    f_unit = np.array([fx, fy])
    magnitude = GG * m1 * m2 / (norm_r**2 + epsilon)
    direction = -np.sign(r)
    return magnitude * f_unit * direction


def RK4Solver(
    t: float,
    r: np.ndarray,
    v: np.ndarray,
    h: float,
    planet: str,
    r_other: np.ndarray,
    v_other: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runge-Kutta 4th order solver for orbital dynamics.
    
    Args:
        t: Current time (years)
        r: Current position vector (AU)
        v: Current velocity vector (AU/year)
        h: Time step (years)
        planet: 'earth' or 'jupiter' to determine dynamics
        r_other: Position of the other planet (AU)
        v_other: Velocity of the other planet (AU/year)
    
    Returns:
        Updated position and velocity vectors
    """
    def acceleration(r_pos: np.ndarray, r_oth: np.ndarray) -> np.ndarray:
        """Calculate total acceleration on the planet."""
        # Force from Sun (at origin)
        r_sun = -r_pos
        f_sun = gravitational_force(MS, ME if planet == 'earth' else MJ, r_sun)
        
        # Force from other planet
        r_rel = r_oth - r_pos
        m_other = MJ if planet == 'earth' else ME
        f_other = gravitational_force(m_other, ME if planet == 'earth' else MJ, r_rel)
        
        mass = ME if planet == 'earth' else MJ
        return (f_sun + f_other) / mass
    
    # k1
    k1_v = acceleration(r, r_other)
    k1_r = v
    
    # k2
    r2 = r + 0.5 * h * k1_r
    v2 = v + 0.5 * h * k1_v
    k2_v = acceleration(r2, r_other)
    k2_r = v2
    
    # k3
    r3 = r + 0.5 * h * k2_r
    v3 = v + 0.5 * h * k2_v
    k3_v = acceleration(r3, r_other)
    k3_r = v3
    
    # k4
    r4 = r + h * k3_r
    v4 = v + h * k3_v
    k4_v = acceleration(r4, r_other)
    k4_r = v4
    
    # Update
    r_new = r + (h / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (h / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return r_new, v_new


def setup_animation() -> Tuple[py.Figure, py.Axes, matplotlib.lines.Line2D, matplotlib.lines.Line2D, py.Text]:
    """
    Setup the animation figure and axes.
    
    Returns:
        Figure, axes, earth line, jupiter line, and text objects
    """
    fig = py.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(-7.2, 7.2)
    ax.set_ylim(-7.2, 7.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Plot Sun at center
    ax.plot(0, 0, 'o', color='orange', markersize=20, label='Sun')
    
    # Initialize trajectory lines
    earth_line, = ax.plot([], [], '-', color='blue', markersize=4, markevery=1, label='Earth')
    jupiter_line, = ax.plot([], [], '-', color='red', markersize=6, markevery=2, label='Super Jupiter')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                        verticalalignment='top', color='white')
    
    return fig, ax, earth_line, jupiter_line, time_text


def animate(i: int, earth_pos: np.ndarray, jupiter_pos: np.ndarray, 
            earth_line: matplotlib.lines.Line2D, jupiter_line: matplotlib.lines.Line2D,
            time_text: py.Text) -> Tuple[matplotlib.lines.Line2D, matplotlib.lines.Line2D, py.Text]:
    """
    Animation update function.
    
    Args:
        i: Current frame index
        earth_pos: Array of Earth positions
        jupiter_pos: Array of Jupiter positions
        earth_line: Line2D object for Earth trajectory
        jupiter_line: Line2D object for Jupiter trajectory
        time_text: Text object for time display
    
    Returns:
        Updated plot objects
    """
    # Earth trail (last 40 steps)
    start_earth = max(0, i - 40)
    earth_line.set_data(earth_pos[start_earth:i+1, 0], earth_pos[start_earth:i+1, 1])
    
    # Jupiter trail (last 200 steps)
    start_jupiter = max(0, i - 200)
    jupiter_line.set_data(jupiter_pos[start_jupiter:i+1, 0], jupiter_pos[start_jupiter:i+1, 1])
    
    # Update time display
    years = i * h
    time_text.set_text(f'Time: {years:.1f} years')
    
    return earth_line, jupiter_line, time_text


def main():
    """Main simulation function."""
    # Initialize arrays
    earth_pos = np.zeros((N, 2))
    earth_vel = np.zeros((N, 2))
    jupiter_pos = np.zeros((N, 2))
    jupiter_vel = np.zeros((N, 2))
    
    # Initial conditions for Earth
    earth_pos[0, 0] = 1.0  # 1 AU on x-axis
    earth_pos[0, 1] = 0.0
    r_earth_init = np.sqrt(earth_pos[0, 0]**2 + earth_pos[0, 1]**2)
    v_earth_mag = np.sqrt(MS * GG / r_earth_init)
    earth_vel[0, 1] = v_earth_mag  # Orbital velocity on y-axis
    
    # Initial conditions for Super Jupiter
    jupiter_pos[0, 0] = 5.2  # 5.2 AU on x-axis
    jupiter_pos[0, 1] = 0.0
    jupiter_vel[0, 1] = 13.06e3 * YEAR / AU  # Velocity in AU/year
    
    # Main simulation loop
    for i in trange(N - 1, desc="Simulating"):
        # Update Earth
        earth_pos[i+1], earth_vel[i+1] = RK4Solver(
            i * h, earth_pos[i], earth_vel[i], h, 'earth',
            jupiter_pos[i], jupiter_vel[i]
        )
        
        # Update Jupiter
        jupiter_pos[i+1], jupiter_vel[i+1] = RK4Solver(
            i * h, jupiter_pos[i], jupiter_vel[i], h, 'jupiter',
            earth_pos[i], earth_vel[i]
        )
    
    # Setup animation
    fig, ax, earth_line, jupiter_line, time_text = setup_animation()
    
    # Add scale bar (1 AU)
    ax.plot([5.5, 6.5], [-6.5, -6.5], 'w-', linewidth=2)
    ax.text(6.0, -6.7, '1 AU', color='white', ha='center', fontsize=10)
    
    # Add labels
    ax.text(0.3, 0.3, 'Sun', color='orange', fontsize=10, fontweight='bold')
    ax.text(earth_pos[0, 0] + 0.2, earth_pos[0, 1] + 0.2, 'Earth', color='blue', fontsize=9)
    ax.text(jupiter_pos[0, 0] + 0.3, jupiter_pos[0, 1] + 0.3, 'Super Jupiter', color='red', fontsize=9)
    
    # Create animation
    anim = matplotlib.animation.FuncAnimation(
        fig, animate, frames=4000, interval=1, blit=False,
        fargs=(earth_pos, jupiter_pos, earth_line, jupiter_line, time_text)
    )
    
    plt.show()


if __name__ == "__main__":
    main()
