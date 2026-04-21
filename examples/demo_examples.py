#!/usr/bin/env python
"""
Exemplos demonstrativos do StellarSeeker.

Scripts prontos para execução que analisam estrelas conhecidas:
- π Mensae (TIC 261136679): Exoplaneta confirmado (~6.26 dias)
- Algol/Beta Persei (TIC 346783960): Binária eclipsante (~2.87 dias)
- TYC 7037-89-1 (TIC 168789840): Sistema sextuplo (~1.57 dias)
"""

import logging
import sys
import os

# Adicionar o diretório pai ao path para importação
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stellar_seeker import StellarSeeker
from stellar_seeker.utils import format_period, btjd_to_datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def analyze_pi_mensae():
    """
    Analisa π Mensae (TIC 261136679), estrela com exoplaneta confirmado.
    
    π Mensae b é um exoplaneta descoberto pelo TESS com período orbital
    de aproximadamente 6.26 dias. Este exemplo demonstra a detecção
    de trânsitos planetários típicos.
    """
    print("\n" + "="*70)
    print("  EXEMPLO 1: π Mensae (TIC 261136679)")
    print("  Exoplaneta confirmado π Mensae b")
    print("="*70)
    
    tic_id = 261136679
    expected_period = 6.26
    
    print(f"\n📊 Iniciando análise de TIC {tic_id}")
    print(f"   Período esperado: ~{expected_period} dias")
    
    try:
        # Inicializar
        seeker = StellarSeeker(tic_id=tic_id)
        
        # Buscar setores
        print("\n🔍 Buscando setores disponíveis...")
        sectors = seeker.search_sectors()
        
        if not sectors:
            print("⚠️ Nenhum setor encontrado. Verifique conexão ou tente outro TIC.")
            return None
        
        print(f"✅ Encontrados {len(sectors)} setores")
        for s in sectors[:3]:  # Mostrar apenas primeiros 3
            print(f"   - Setor {s['sector']}: {s['duration']:.1f} dias")
        
        # Baixar dados
        print("\n📥 Baixando curva de luz...")
        lc = seeker.download_lightcurve(sector_index=0)
        print(f"✅ Download concluído: {len(lc)} pontos de dados")
        
        # Calcular periodograma
        print("\n📈 Calculando periodograma...")
        pg = seeker.calculate_periodogram(
            minimum_period=0.5,
            maximum_period=15
        )
        
        period = seeker.get_primary_period()
        print(f"🎯 Período detectado: {period:.4f} dias")
        print(f"   ({format_period(period)})")
        
        # Comparar com período esperado
        if period:
            diff = abs(period - expected_period) / expected_period * 100
            print(f"   Diferença do esperado: {diff:.2f}%")
        
        # Dobrar fase
        print("\n🔄 Dobrando fase...")
        fig, phase, flux = seeker.fold_phase()
        print("✅ Fase dobrada com sucesso")
        
        # Detectar trânsitos
        print("\n🔎 Detectando trânsitos...")
        transits = seeker.detect_transits(min_depth=0.0005)
        print(f"✅ {len(transits)} trânsitos candidatos detectados")
        
        if transits:
            print("\nPrincipais trânsitos:")
            for i, t in enumerate(transits[:3], 1):
                dt = btjd_to_datetime(t['time_center'])
                print(f"  {i}. {dt.strftime('%Y-%m-%d %H:%M')} - "
                      f"Profundidade: {t['depth']:.4f}, SNR: {t['snr']:.1f}")
        
        return seeker
        
    except Exception as e:
        print(f"\n❌ Erro na análise: {str(e)}")
        return None


def analyze_algol():
    """
    Analisa Algol/Beta Persei (TIC 346783960), binária eclipsante famosa.
    
    Algol é uma das primeiras variáveis eclipsantes descobertas,
    com período de aproximadamente 2.87 dias. Demonstra eclipses
    estelares profundos em contraste com trânsitos planetários.
    """
    print("\n" + "="*70)
    print("  EXEMPLO 2: Algol / Beta Persei (TIC 346783960)")
    print("  Binária eclipsante clássica")
    print("="*70)
    
    tic_id = 346783960
    expected_period = 2.87
    
    print(f"\n📊 Iniciando análise de TIC {tic_id}")
    print(f"   Período esperado: ~{expected_period} dias")
    
    try:
        seeker = StellarSeeker(tic_id=tic_id)
        
        print("\n🔍 Buscando setores...")
        sectors = seeker.search_sectors()
        
        if not sectors:
            print("⚠️ Nenhum setor encontrado.")
            return None
        
        print(f"✅ {len(sectors)} setores disponíveis")
        
        print("\n📥 Baixando dados...")
        lc = seeker.download_lightcurve(sector_index=0)
        print(f"✅ {len(lc)} pontos baixados")
        
        print("\n📈 Calculando periodograma...")
        pg = seeker.calculate_periodogram(
            minimum_period=1,
            maximum_period=10
        )
        
        period = seeker.get_primary_period()
        print(f"🎯 Período detectado: {period:.4f} dias")
        print(f"   ({format_period(period)})")
        
        if period:
            diff = abs(period - expected_period) / expected_period * 100
            print(f"   Diferença do esperado: {diff:.2f}%")
        
        print("\n🔄 Dobrando fase...")
        fig, phase, flux = seeker.fold_phase()
        print("✅ Fase dobrada")
        
        # Para binárias eclipsantes, usar profundidade maior
        print("\n🔎 Detectando eclipses...")
        transits = seeker.detect_transits(min_depth=0.01)
        print(f"✅ {len(transits)} eclipses detectados")
        
        return seeker
        
    except Exception as e:
        print(f"\n❌ Erro na análise: {str(e)}")
        return None


def analyze_tyc_7037():
    """
    Analisa TYC 7037-89-1 (TIC 168789840), sistema sextuplo raro.
    
    Este é um sistema estelar excepcional com seis estrelas,
    demonstrando a versatilidade do StellarSeeker para sistemas complexos.
    """
    print("\n" + "="*70)
    print("  EXEMPLO 3: TYC 7037-89-1 (TIC 168789840)")
    print("  Sistema estelar sextuplo")
    print("="*70)
    
    tic_id = 168789840
    expected_period = 1.57
    
    print(f"\n📊 Iniciando análise de TIC {tic_id}")
    print(f"   Período principal esperado: ~{expected_period} dias")
    
    try:
        seeker = StellarSeeker(tic_id=tic_id)
        
        print("\n🔍 Buscando setores...")
        sectors = seeker.search_sectors()
        
        if not sectors:
            print("⚠️ Nenhum setor encontrado.")
            return None
        
        print(f"✅ {len(sectors)} setores disponíveis")
        
        print("\n📥 Baixando dados...")
        lc = seeker.download_lightcurve(sector_index=0)
        print(f"✅ {len(lc)} pontos baixados")
        
        print("\n📈 Calculando periodograma...")
        pg = seeker.calculate_periodogram(
            minimum_period=0.5,
            maximum_period=5
        )
        
        period = seeker.get_primary_period()
        print(f"🎯 Período detectado: {period:.4f} dias")
        print(f"   ({format_period(period)})")
        
        if period:
            diff = abs(period - expected_period) / expected_period * 100
            print(f"   Diferença do esperado: {diff:.2f}%")
        
        print("\n🔄 Dobrando fase...")
        fig, phase, flux = seeker.fold_phase()
        print("✅ Fase dobrada")
        
        print("\n🔎 Detectando eventos...")
        transits = seeker.detect_transits(min_depth=0.001)
        print(f"✅ {len(transits)} eventos detectados")
        
        return seeker
        
    except Exception as e:
        print(f"\n❌ Erro na análise: {str(e)}")
        return None


def run_all_examples():
    """Executa todos os exemplos demonstrativos sequencialmente."""
    print("\n" + "#"*70)
    print("#  STELLARSEEKER - EXEMPLOS DEMONSTRATIVOS")
    print("#  Análise de estrelas conhecidas com dados do TESS")
    print("#"*70)
    
    results = {}
    
    # Exemplo 1: π Mensae
    print("\n\n")
    results['pi_mensae'] = analyze_pi_mensae()
    
    # Exemplo 2: Algol
    print("\n\n")
    results['algol'] = analyze_algol()
    
    # Exemplo 3: TYC 7037
    print("\n\n")
    results['tyc_7037'] = analyze_tyc_7037()
    
    # Resumo final
    print("\n" + "="*70)
    print("  RESUMO DOS RESULTADOS")
    print("="*70)
    
    for name, seeker in results.items():
        if seeker:
            period = seeker.get_primary_period()
            if period:
                print(f"\n✓ {name.upper()}")
                print(f"  TIC: {seeker.tic_id}")
                print(f"  Período detectado: {period:.4f} dias")
                print(f"  Trânsitos/Eclipses: {len(seeker.transits)}")
        else:
            print(f"\n✗ {name.upper()}: Análise não concluída")
    
    print("\n" + "="*70)
    print("  Exemplos concluídos!")
    print("  Para mais informações, consulte a documentação no Jupyter Notebook.")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Executar exemplos demonstrativos do StellarSeeker"
    )
    parser.add_argument(
        '--example', '-e',
        choices=['pi_mensae', 'algol', 'tyc_7037', 'all'],
        default='all',
        help='Qual exemplo executar (padrão: all)'
    )
    parser.add_argument(
        '--export', '-x',
        action='store_true',
        help='Exportar resultados para arquivos'
    )
    
    args = parser.parse_args()
    
    if args.example == 'all':
        results = run_all_examples()
        
        # Exportar se solicitado
        if args.export:
            print("\n💾 Exportando resultados...")
            for name, seeker in results.items():
                if seeker and seeker.lightcurve is not None:
                    output_dir = f"./results_{name}"
                    files = seeker.export_results(output_dir=output_dir)
                    print(f"  {name}: {len(files)} arquivos salvos em {output_dir}/")
    
    elif args.example == 'pi_mensae':
        analyze_pi_mensae()
    
    elif args.example == 'algol':
        analyze_algol()
    
    elif args.example == 'tyc_7037':
        analyze_tyc_7037()
