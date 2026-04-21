"""
Testes para o módulo de cálculos astronômicos.
"""

import numpy as np
import pytest
from astropy import units as u

from astropyflow.core.calculations import (
    calculate_angular_distance,
    calculate_absolute_magnitude,
    calculate_luminosity,
    convert_units,
    parsec_to_lightyear,
    magnitude_to_flux,
    flux_to_magnitude,
)


class TestAngularDistance:
    """Testes para cálculo de distância angular."""

    def test_zero_distance(self):
        """Testa distância zero entre pontos iguais."""
        dist = calculate_angular_distance(0, 0, 0, 0)
        assert dist.value == pytest.approx(0.0, abs=1e-10)

    def test_one_degree_separation(self):
        """Testa separação de 1 grau em RA."""
        dist = calculate_angular_distance(0, 0, 1, 0)
        assert dist.value == pytest.approx(1.0, rel=1e-6)

    def test_ninety_degree_separation(self):
        """Testa separação de 90 graus."""
        dist = calculate_angular_distance(0, 0, 90, 0)
        assert dist.value == pytest.approx(90.0, rel=1e-6)

    def test_array_input(self):
        """Testa entrada em array (vetorização)."""
        ra1 = [0, 0, 0]
        dec1 = [0, 0, 0]
        ra2 = [1, 2, 3]
        dec2 = [0, 0, 0]

        dist = calculate_angular_distance(ra1, dec1, ra2, dec2)
        assert len(dist) == 3
        assert dist[0].value == pytest.approx(1.0, rel=1e-6)
        assert dist[1].value == pytest.approx(2.0, rel=1e-6)
        assert dist[2].value == pytest.approx(3.0, rel=1e-6)

    def test_different_units(self):
        """Testa conversão para diferentes unidades."""
        dist_deg = calculate_angular_distance(0, 0, 1, 0, unit=u.deg)
        dist_rad = calculate_angular_distance(0, 0, 1, 0, unit=u.rad)

        # 1 grau = π/180 radianos
        expected_rad = np.pi / 180
        assert dist_rad.value == pytest.approx(expected_rad, rel=1e-6)


class TestAbsoluteMagnitude:
    """Testes para magnitude absoluta."""

    def test_at_10_parsecs(self):
        """Testa que a 10 pc, m = M."""
        M = calculate_absolute_magnitude(5.0, 10)
        assert M == pytest.approx(5.0, rel=1e-10)

    def test_at_100_parsecs(self):
        """Testa módulo de distância a 100 pc."""
        # m - M = 5*log10(100) - 5 = 5
        M = calculate_absolute_magnitude(10.0, 100)
        assert M == pytest.approx(5.0, rel=1e-10)

    def test_with_quantity(self):
        """Testa com Quantity do Astropy."""
        M = calculate_absolute_magnitude(5.0, 10 * u.pc)
        assert M == pytest.approx(5.0, rel=1e-10)

    def test_array_input(self):
        """Testa entrada em array."""
        mags = [5.0, 10.0, 15.0]
        dists = [10, 100, 1000]

        M = calculate_absolute_magnitude(mags, dists)
        assert len(M) == 3
        # Todos devem ter M = m - 5*log10(d/10)
        expected = np.array([5.0, 5.0, 5.0])
        assert np.allclose(M, expected)


class TestLuminosity:
    """Testes para cálculo de luminosidade."""

    def test_sun_at_10pc(self):
        """Testa luminosidade solar a 10 pc."""
        # Sol a 10 pc teria m ≈ 4.74 (magnitude absoluta)
        L = calculate_luminosity(4.74, 10 * u.pc)
        # Deve ser próximo de 1 L_sun
        assert L == pytest.approx(1.0, rel=0.1)

    def test_brighter_star(self):
        """Testa estrela mais brilhante que o Sol."""
        # Estrela com M = 0 deve ser mais luminosa
        L = calculate_luminosity(0.0, 10 * u.pc)
        assert L > 10  # Muito mais luminosa que o Sol

    def test_fainter_star(self):
        """Testa estrela mais fraca que o Sol."""
        # Estrela com M = 10 deve ser menos luminosa
        L = calculate_luminosity(10.0, 10 * u.pc)
        assert L < 0.01  # Muito menos luminosa que o Sol


class TestUnitConversion:
    """Testes para conversão de unidades."""

    def test_parsec_to_lightyear(self):
        """Testa conversão pc -> ly."""
        from astropy import units as u
        ly = parsec_to_lightyear(1.0)
        assert ly == pytest.approx(3.26156, rel=1e-4)

    def test_convert_units_custom(self):
        """Testa conversão genérica."""
        from astropy import units as u
        result = convert_units(100, u.pc, u.lightyear)
        assert result.value == pytest.approx(326.156, rel=1e-3)

    def test_convert_with_quantity(self):
        """Testa conversão mantendo Quantity."""
        from astropy import units as u
        val = 100 * u.pc
        result = convert_units(val, u.pc, u.lightyear)
        assert isinstance(result, u.Quantity)
        assert result.unit == u.lightyear


class TestMagnitudeFluxConversion:
    """Testes para conversão magnitude-fluxo."""

    def test_zero_magnitude(self):
        """Testa magnitude 0 -> fluxo zero_point."""
        flux = magnitude_to_flux(0.0)
        assert flux == pytest.approx(3631.0, rel=1e-10)  # Jy

    def test_flux_to_magnitude_roundtrip(self):
        """Testa conversão ida e volta."""
        mag = 5.0
        flux = magnitude_to_flux(mag)
        mag_back = flux_to_magnitude(flux)
        assert mag_back == pytest.approx(mag, rel=1e-10)

    def test_different_bands(self):
        """Testa diferentes bandas fotométricas."""
        flux_V = magnitude_to_flux(0.0, band="V")
        flux_B = magnitude_to_flux(0.0, band="B")

        # Bandas diferentes têm zero points diferentes
        assert flux_V != flux_B
        assert flux_V == pytest.approx(3631.0, rel=1e-10)
        assert flux_B == pytest.approx(4260.0, rel=1e-10)
