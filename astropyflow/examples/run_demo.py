#!/usr/bin/env python
"""
Script de demonstração do AstroPyFlow.

Executa exemplos completos de:
1. Plot da posição do Sol
2. Simulação orbital básica (3 corpos)
3. Validação de dados + cálculo de luminosidade
4. Curva de luz de exoplaneta
5. Classificação de exoplanetas

Uso:
    python examples/run_demo.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para servidores
import matplotlib.pyplot as plt
import numpy as np

# Adicionar src ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astropyflow.core.data_handler import create_mock_catalog, validate_catalog
from astropyflow.core.calculations import calculate_luminosity, calculate_angular_distance
from astropyflow.viz.plotting import (
    plot_sun_position,
    plot_light_curve,
    generate_mock_light_curve,
    save_plot,
)
from astropyflow.sim.solar_system import simulate_orbits, plot_orbits
from astropyflow.sim.exoplanets import (
    classify_exoplanet,
    is_in_habitable_zone,
    generate_mock_exoplanets,
    analyze_exoplanet_system,
)
from astropyflow.utils.logger import setup_logger


def main():
    """Executa todas as demonstrações."""
    # Configurar logger
    logger = setup_logger("astropyflow_demo", level=logging.INFO)
    logger.info("=" * 60)
    logger.info("ASTROPYFLOW - DEMONSTRAÇÃO COMPLETA")
    logger.info("=" * 60)

    # Criar diretório de saída
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diretório de saída: {output_dir}")

    results = {}

    # =========================================================================
    # 1. POSIÇÃO DO SOL
    # =========================================================================
    logger.info("\n[1/5] Plot da Posição do Sol")
    logger.info("-" * 40)

    try:
        fig_sun, ax_sun = plot_sun_position(
            location="Sao Paulo",
            date="2024-06-21",  # Solstício de inverno
            output_path=output_dir / "sun_position.png",
        )
        plt.close(fig_sun)
        results["sun_position"] = "OK"
        logger.info("✓ Gráfico de posição solar salvo")
    except Exception as e:
        logger.error(f"✗ Erro no plot do Sol: {e}")
        results["sun_position"] = f"ERROR: {e}"

    # =========================================================================
    # 2. SIMULAÇÃO ORBITAL
    # =========================================================================
    logger.info("\n[2/5] Simulação Orbital (Sistema Solar)")
    logger.info("-" * 40)

    try:
        # Simular Sistema Solar interno (Sol + Mercúrio + Vênus + Terra)
        sim_result = simulate_orbits(
            n_bodies=4,
            steps=500,
            dt_days=2.0,
            seed=42,
            include_sun=True,
        )

        info = sim_result["info"]
        logger.info(f"  Corpos simulados: {info['n_bodies']}")
        logger.info(f"  Passos de integração: {info['steps']}")
        logger.info(f"  Tempo total: {info['total_years']:.2f} anos")

        # Plotar órbitas
        fig_orbit, ax_orbit = plot_orbits(
            sim_result["trajectories"],
            sim_result["bodies"],
            output_path=str(output_dir / "orbits.png"),
            scale_au=True,
        )
        plt.close(fig_orbit)

        results["orbit_simulation"] = "OK"
        logger.info("✓ Simulação orbital concluída e plot salva")
    except Exception as e:
        logger.error(f"✗ Erro na simulação orbital: {e}")
        results["orbit_simulation"] = f"ERROR: {e}"

    # =========================================================================
    # 3. VALIDAÇÃO DE DADOS + CÁLCULO DE LUMINOSIDADE
    # =========================================================================
    logger.info("\n[3/5] Validação de Dados e Cálculo de Luminosidade")
    logger.info("-" * 40)

    try:
        # Criar catálogo mock
        catalog = create_mock_catalog(n_stars=200, seed=123)
        logger.info(f"  Catálogo criado: {len(catalog)} estrelas")

        # Validar catálogo
        validation = validate_catalog(catalog)
        logger.info(f"  Linhas originais: {validation['original_rows']}")
        logger.info(f"  Linhas após limpeza: {validation['cleaned_rows']}")
        logger.info(f"  Válidas: {validation['is_valid']}")

        # Calcular luminosidade para algumas estrelas
        clean_data = validation["cleaned_data"]
        luminosities = []

        for idx in range(min(10, len(clean_data))):
            row = clean_data.iloc[idx]
            L = calculate_luminosity(
                apparent_mag=row["apparent_mag"],
                distance=row["distance_pc"],
            )
            luminosities.append({
                "star": row["star_id"],
                "L/L_sun": L,
                "spectral_class": row["spectral_class"],
            })

        logger.info("  Luminosidades calculadas (primeiras 10 estrelas):")
        for lum in luminosities[:5]:
            logger.info(f"    {lum['star']}: {lum['L/L_sun']:.2e} L_sun ({lum['spectral_class']})")

        # Calcular distância angular entre duas estrelas
        if len(clean_data) >= 2:
            star1 = clean_data.iloc[0]
            star2 = clean_data.iloc[1]
            ang_dist = calculate_angular_distance(
                star1["ra"], star1["dec"],
                star2["ra"], star2["dec"],
            )
            logger.info(f"  Distância angular entre primeiras 2 estrelas: {ang_dist:.2f}")

        results["data_validation"] = "OK"
        logger.info("✓ Validação e cálculos concluídos")
    except Exception as e:
        logger.error(f"✗ Erro na validação/cálculos: {e}")
        results["data_validation"] = f"ERROR: {e}"

    # =========================================================================
    # 4. CURVA DE LUZ DE EXOPLANETA
    # =========================================================================
    logger.info("\n[4/5] Curva de Luz de Exoplaneta")
    logger.info("-" * 40)

    try:
        # Gerar curva de luz mock
        time, flux, error = generate_mock_light_curve(
            n_points=500,
            period=3.5,
            transit_depth=0.015,
            transit_duration=0.12,
            noise_level=0.0008,
            seed=42,
        )

        # Identificar centro do trânsito
        transit_center = 3.5 / 2  # Meio do primeiro período

        # Plotar
        fig_lc, ax_lc = plot_light_curve(
            time, flux, error,
            title="Curva de Luz Simulada - Exoplaneta em Trânsito",
            show_transit=True,
            transit_center=transit_center,
            transit_depth=0.015,
            output_path=output_dir / "light_curve.png",
        )
        plt.close(fig_lc)

        results["light_curve"] = "OK"
        logger.info(f"  Pontos de dados: {len(time)}")
        logger.info(f"  Profundidade do trânsito: 1.5%")
        logger.info("✓ Curva de luz plotada e salva")
    except Exception as e:
        logger.error(f"✗ Erro na curva de luz: {e}")
        results["light_curve"] = f"ERROR: {e}"

    # =========================================================================
    # 5. CLASSIFICAÇÃO DE EXOPLANETAS
    # =========================================================================
    logger.info("\n[5/5] Classificação de Exoplanetas e Zona Habitável")
    logger.info("-" * 40)

    try:
        # Gerar exoplanetas mock
        planets = generate_mock_exoplanets(n_planets=8, seed=42)

        # Analisar sistema
        df_analysis = analyze_exoplanet_system(
            planets,
            star_temp=5778,  # Tipo solar
            star_radius=1.0,
        )

        logger.info("  Exoplanetas classificados:")
        for _, row in df_analysis.iterrows():
            hab_status = "🟢 HABITÁVEL" if row["in_habitable_zone"] else "⭕ Não habitável"
            logger.info(
                f"    {row['name']}: R={row['radius_earth']:.2f} R⊕, "
                f"T={row['temp_k']:.0f}K, {row['class']:<15} {hab_status}"
            )

        # Verificar zona habitável detalhada
        earth_like = planets[2]  # Terceiro planeta (índice 2)
        if earth_like.semi_major_axis:
            hz_check = is_in_habitable_zone(
                semi_major_axis=earth_like.semi_major_axis,
                star_temp=5778,
                star_radius=1.0,
                eccentricity=earth_like.eccentricity,
            )
            logger.info(f"\n  Análise detalhada de {earth_like.name}:")
            logger.info(f"    Na zona habitável: {hz_check['in_zone']}")
            logger.info(f"    Classe: {hz_check['zone_class']}")
            logger.info(f"    Fluxo recebido: {hz_check['flux_received']:.2f} x Terra")

        results["exoplanet_classification"] = "OK"
        logger.info("✓ Classificação de exoplanetas concluída")
    except Exception as e:
        logger.error(f"✗ Erro na classificação: {e}")
        results["exoplanet_classification"] = f"ERROR: {e}"

    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("RESUMO DA DEMONSTRAÇÃO")
    logger.info("=" * 60)

    all_ok = True
    for test, result in results.items():
        status = "✓" if result == "OK" else "✗"
        logger.info(f"  {status} {test}: {result}")
        if result != "OK":
            all_ok = False

    logger.info("")
    if all_ok:
        logger.info("🎉 TODAS AS DEMONSTRAÇÕES CONCLUÍDAS COM SUCESSO!")
    else:
        logger.info("⚠️ Algumas demonstrações falharam. Verifique os logs.")

    logger.info(f"\nArquivos de saída em: {output_dir.absolute()}")
    logger.info("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
