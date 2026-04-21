"""
Módulo principal do StellarSeeker.

Contém a classe StellarSeeker que encapsula todas as funcionalidades
para análise de curvas de luz e detecção de exoplanetas.
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import lightkurve as lk
from scipy import signal, integrate, interpolate
from scipy.stats import sigmaclip

# Configuração do logger
logger = logging.getLogger(__name__)


# Type hint para Periodogram (pode variar entre versões do lightkurve)
try:
    PeriodogramType = lk.Periodogram
except AttributeError:
    PeriodogramType = object


class StellarSeeker:
    """
    Classe principal para análise de curvas de luz astronômicas e
    descoberta de exoplanetas usando dados da missão TESS.
    
    Esta classe fornece uma interface completa para baixar, visualizar
    e analisar curvas de luz de estrelas do catálogo TESS (TIC), permitindo
    a detecção de trânsitos planetários e eclipses estelares.
    
    Attributes:
        tic_id (int): Identificador TIC (TESS Input Catalog) da estrela alvo.
        sector (int, optional): Setor TESS específico para análise.
        author (str): Autor dos dados (padrão 'SPOC').
        lightcurve (LightCurve): Objeto LightCurve contendo os dados baixados.
        periodogram (Periodogram): Objeto Periodogram calculado.
        primary_period (float): Período fundamental detectado em dias.
        transits (list): Lista de trânsitos detectados.
        
    Example:
        >>> seeker = StellarSeeker(tic_id=261136679)
        >>> sectors = seeker.search_sectors()
        >>> lc = seeker.download_lightcurve(sector_index=0)
        >>> seeker.plot_lightcurve()
        >>> period = seeker.get_primary_period()
        >>> seeker.fold_phase()
    """
    
    def __init__(self, tic_id: int, sector: Optional[int] = None, 
                 author: str = 'SPOC'):
        """
        Inicializa uma instância do StellarSeeker.
        
        Args:
            tic_id (int): Identificador TIC da estrela alvo.
            sector (int, optional): Setor TESS específico. Se None, usa o primeiro disponível.
            author (str): Autor dos dados. Padrão é 'SPOC' (Science Processing Operations Center).
            
        Raises:
            ValueError: Se o tic_id não for um inteiro positivo.
        """
        if not isinstance(tic_id, int) or tic_id <= 0:
            raise ValueError("tic_id deve ser um inteiro positivo")
        
        self.tic_id = tic_id
        self.sector = sector
        self.author = author
        self.lightcurve = None
        self.periodogram = None
        self.primary_period = None
        self.transits = []
        self._search_results = None
        
        # Configurar cache local
        self._cache_dir = os.path.expanduser('~/.stellar_seeker/cache')
        os.makedirs(self._cache_dir, exist_ok=True)
        
        logger.info(f"StellarSeeker inicializado para TIC {tic_id}")
    
    def search_sectors(self) -> List[Dict[str, Any]]:
        """
        Busca todos os setores TESS disponíveis para o TIC fornecido.
        
        Este método consulta o arquivo MAST para encontrar todas as observações
        disponíveis da estrela especificada, retornando informações sobre
        cada setor incluindo índice, nome, duração e data de observação.
        
        Returns:
            list: Lista de dicionários com informações dos setores.
                  Cada dicionário contém:
                  - 'index': Índice para seleção
                  - 'sector': Número do setor
                  - 'observation': Nome da observação
                  - 'duration': Duração em dias
                  - 'target_name': Nome do alvo
                  
        Raises:
            ConnectionError: Se não houver conexão com o servidor MAST.
            ValueError: Se nenhum setor for encontrado para o TIC.
            
        Example:
            >>> seeker = StellarSeeker(tic_id=261136679)
            >>> sectors = seeker.search_sectors()
            >>> for s in sectors:
            ...     print(f"Setor {s['sector']}: {s['duration']} dias")
        """
        logger.info(f"Buscando setores para TIC {self.tic_id}")
        
        try:
            self._search_results = lk.search_lightcurve(
                f"TIC {self.tic_id}",
                author=self.author
            )
            
            if len(self._search_results) == 0:
                logger.warning(f"Nenhum setor encontrado para TIC {self.tic_id}")
                return []
            
            sectors_info = []
            for idx, row in enumerate(self._search_results):
                sector_info = {
                    'index': idx,
                    'sector': row.get('sector', 'N/A'),
                    'observation': row.get('observation', ''),
                    'duration': row.get('duration', 0),
                    'target_name': row.get('target_name', f'TIC {self.tic_id}')
                }
                sectors_info.append(sector_info)
            
            logger.info(f"Encontrados {len(sectors_info)} setores")
            return sectors_info
            
        except Exception as e:
            logger.error(f"Erro ao buscar setores: {str(e)}")
            raise ConnectionError(f"Falha ao conectar com MAST: {str(e)}")
    
    def download_lightcurve(self, sector_index: int = 0) -> lk.LightCurve:
        """
        Baixa os dados da curva de luz para o setor especificado.
        
        Este método recupera os dados de fotometria do arquivo MAST,
        incluindo tempo (BTJD), fluxo, erro de fluxo e qualidade dos dados.
        Os dados são armazenados em cache local para evitar re-download.
        
        Args:
            sector_index (int): Índice do setor na lista de resultados da busca.
                               Padrão é 0 (primeiro setor disponível).
                               
        Returns:
            LightCurve: Objeto LightCurve do lightkurve contendo os dados.
            
        Raises:
            IndexError: Se o índice do setor estiver fora do intervalo válido.
            RuntimeError: Se a busca de setores não foi realizada antes.
            ConnectionError: Se falhar ao baixar os dados.
            
        Example:
            >>> seeker = StellarSeeker(tic_id=261136679)
            >>> seeker.search_sectors()
            >>> lc = seeker.download_lightcurve(sector_index=0)
            >>> print(f"Dados baixados: {len(lc)} pontos")
        """
        if self._search_results is None:
            logger.info("Realizando busca automática de setores...")
            self.search_sectors()
        
        if self._search_results is None or len(self._search_results) == 0:
            raise RuntimeError("Nenhum setor disponível. Execute search_sectors() primeiro.")
        
        if sector_index < 0 or sector_index >= len(self._search_results):
            raise IndexError(
                f"Índice {sector_index} inválido. Disponível: 0-{len(self._search_results)-1}"
            )
        
        logger.info(f"Baixando dados do setor {sector_index}...")
        
        try:
            self.lightcurve = self._search_results[sector_index].download()
            
            # Remover dados com qualidade ruim
            self.lightcurve = self.lightcurve.remove_nans()
            
            logger.info(f"Dados baixados: {len(self.lightcurve)} pontos de tempo")
            return self.lightcurve
            
        except Exception as e:
            logger.error(f"Erro ao baixar curva de luz: {str(e)}")
            raise ConnectionError(f"Falha ao baixar dados: {str(e)}")
    
    def plot_lightcurve(self, normalize: bool = True, color: str = 'blue',
                        marker: str = '.', linewidth: float = 0,
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Gera e exibe um gráfico da curva de luz.
        
        Cria uma visualização profissional da curva de luz com opções
        de personalização para cores, marcadores e normalização.
        
        Args:
            normalize (bool): Se True, normaliza o fluxo dividindo pela mediana.
                             Padrão é True.
            color (str): Cor dos pontos/linha. Padrão é 'blue'.
            marker (str): Estilo do marcador. Padrão é '.'.
            linewidth (float): Espessura da linha. 0 significa sem linha.
            figsize (tuple): Tamanho da figura (largura, altura) em polegadas.
            save_path (str, optional): Caminho para salvar a figura. Se None, não salva.
            
        Returns:
            Figure: Objeto matplotlib Figure contendo o gráfico.
            
        Raises:
            RuntimeError: Se nenhuma curva de luz foi carregada.
            
        Example:
            >>> seeker.plot_lightcurve(normalize=True, color='red')
            >>> seeker.plot_lightcurve(save_path='lightcurve.png')
        """
        if self.lightcurve is None:
            raise RuntimeError("Nenhuma curva de luz carregada. Execute download_lightcurve() primeiro.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        time = self.lightcurve.time.value
        flux = self.lightcurve.flux.value
        
        if normalize:
            flux_median = np.median(flux)
            flux = flux / flux_median
            ylabel = 'Fluxo Normalizado'
        else:
            ylabel = 'Fluxo (e⁻/s)'
        
        if linewidth > 0:
            ax.plot(time, flux, color=color, linewidth=linewidth, marker=marker, 
                   markersize=3, alpha=0.7)
        else:
            ax.scatter(time, flux, c=color, s=10, marker=marker, alpha=0.5)
        
        ax.set_xlabel('Tempo (BTJD)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Curva de Luz - TIC {self.tic_id}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Adicionar informações sobre o setor
        if self.sector:
            ax.text(0.02, 0.98, f'Setor: {self.sector}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Gráfico salvo em {save_path}")
        
        return fig
    
    def calculate_periodogram(self, oversample_factor: int = 10,
                              minimum_period: float = 0.1,
                              maximum_period: float = 30,
                              plot: bool = False) -> Any:
        """
        Calcula o periodograma da curva de luz usando o método de Lomb-Scargle.
        
        O periodograma identifica periodicidades nos dados, essencial para
        detectar períodos orbitais de exoplanetas ou períodos de rotação estelar.
        
        Args:
            oversample_factor (int): Fator de sobreamostragem para maior resolução.
                                    Padrão é 10.
            minimum_period (float): Período mínimo a buscar em dias. Padrão é 0.1.
            maximum_period (float): Período máximo a buscar em dias. Padrão é 30.
            plot (bool): Se True, plota o periodograma automaticamente.
            
        Returns:
            Periodogram: Objeto Periodogram do lightkurve.
            
        Raises:
            RuntimeError: Se nenhuma curva de luz foi carregada.
            
        Example:
            >>> pg = seeker.calculate_periodogram()
            >>> period = seeker.get_primary_period()
            >>> print(f"Período candidato: {period:.4f} dias")
        """
        if self.lightcurve is None:
            raise RuntimeError("Nenhuma curva de luz carregada. Execute download_lightcurve() primeiro.")
        
        logger.info("Calculando periodograma...")
        
        try:
            self.periodogram = self.lightcurve.to_periodogram(
                minimum_period=minimum_period * u.day,
                maximum_period=maximum_period * u.day,
                oversample_factor=oversample_factor
            )
            
            self.primary_period = self.periodogram.period_at_max_power.value
            
            logger.info(f"Período de máxima potência: {self.primary_period:.6f} dias")
            
            if plot:
                self._plot_periodogram()
            
            return self.periodogram
            
        except Exception as e:
            logger.error(f"Erro ao calcular periodograma: {str(e)}")
            raise
    
    def _plot_periodogram(self, figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
        """
        Plota o periodograma destacando o pico de máxima potência.
        
        Args:
            figsize (tuple): Tamanho da figura.
            
        Returns:
            Figure: Objeto matplotlib Figure.
        """
        if self.periodogram is None:
            raise RuntimeError("Nenhum periodograma calculado.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        period = self.periodogram.period.value
        power = self.periodogram.power.value
        
        ax.semilogx(period, power, 'b-', linewidth=1, alpha=0.7)
        
        # Destacar o pico
        max_period = self.primary_period
        max_power = np.max(power)
        ax.axvline(x=max_period, color='red', linestyle='--', 
                  label=f'Pico: {max_period:.4f} dias')
        ax.plot(max_period, max_power, 'ro', markersize=10)
        
        ax.set_xlabel('Período (dias)', fontsize=12)
        ax.set_ylabel('Potência (e⁻/s)', fontsize=12)
        ax.set_title(f'Periodograma - TIC {self.tic_id}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_primary_period(self) -> Optional[float]:
        """
        Extrai e retorna o período fundamental detectado no periodograma.
        
        Returns:
            float or None: Período em dias correspondente ao pico de máxima
                          potência, ou None se nenhum periodograma foi calculado.
                          
        Example:
            >>> period = seeker.get_primary_period()
            >>> if period:
            ...     print(f"Período candidato: {period:.4f} dias")
        """
        if self.periodogram is None:
            logger.warning("Nenhum periodograma calculado. Execute calculate_periodogram() primeiro.")
            return None
        
        return self.primary_period
    
    def fold_phase(self, period: Optional[float] = None, wrap_phase: float = 2,
                   eps_correction: float = 0.002,
                   figsize: Tuple[int, int] = (10, 6),
                   save_path: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
        """
        Realiza a dobradura de fase (phase folding) da curva de luz.
        
        A dobradura de fase sobrepõe múltiplos ciclos da curva de luz,
        facilitando a visualização de padrões periódicos como trânsitos
        planetários ou eclipses estelares.
        
        Args:
            period (float, optional): Período para dobradura em dias.
                                     Se None, usa o período do periodograma.
            wrap_phase (float): Quantas vezes repetir o ciclo no gráfico.
                               Padrão é 2 para mostrar continuidade.
            eps_correction (float): Correção fina manual do período.
            figsize (tuple): Tamanho da figura.
            save_path (str, optional): Caminho para salvar a figura.
            
        Returns:
            tuple: (Figure, phase, flux) onde:
                   - Figure: Objeto matplotlib Figure
                   - phase: Array de fases (0 a wrap_phase)
                   - flux: Array de fluxos correspondentes
                   
        Raises:
            RuntimeError: Se nenhuma curva de luz foi carregada.
            ValueError: Se nenhum período foi fornecido ou calculado.
            
        Example:
            >>> seeker.calculate_periodogram()
            >>> fig, phase, flux = seeker.fold_phase()
            >>> # Ou com período manual
            >>> fig, phase, flux = seeker.fold_phase(period=6.26)
        """
        if self.lightcurve is None:
            raise RuntimeError("Nenhuma curva de luz carregada.")
        
        if period is None:
            period = self.get_primary_period()
            if period is None:
                raise ValueError(
                    "Nenhum período fornecido e nenhum periodograma calculado. "
                    "Forneça um período ou execute calculate_periodogram() primeiro."
                )
        
        # Aplicar correção se necessário
        effective_period = period * (1 + eps_correction)
        
        logger.info(f"Dobrando fase com período {effective_period:.6f} dias")
        
        # Calcular fase
        time = self.lightcurve.time.value
        flux = self.lightcurve.flux.value
        
        # Normalizar pelo tempo zero
        t0 = np.min(time)
        phase = ((time - t0) % effective_period) / effective_period
        
        # Ordenar por fase para plotagem
        sort_idx = np.argsort(phase)
        phase_sorted = phase[sort_idx]
        flux_sorted = flux[sort_idx]
        
        # Normalizar fluxo
        flux_normalized = flux_sorted / np.median(flux_sorted)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(phase_sorted, flux_normalized, c='blue', s=5, alpha=0.5, marker='.')
        
        # Plotar versão repetida para wrap_phase > 1
        if wrap_phase > 1:
            for i in range(1, int(wrap_phase)):
                ax.scatter(phase_sorted + i, flux_normalized, c='blue', s=5, alpha=0.3, marker='.')
        
        ax.set_xlabel('Fase', fontsize=12)
        ax.set_ylabel('Fluxo Normalizado', fontsize=12)
        ax.set_title(
            f'Curva de Luz Dobrada - TIC {self.tic_id}\nPeríodo: {effective_period:.6f} dias',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlim(0, wrap_phase)
        ax.grid(True, alpha=0.3)
        
        # Marcar trânsito potencial em fase 0 e 1
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Gráfico de fase salvo em {save_path}")
        
        return fig, phase_sorted, flux_normalized
    
    def detect_transits(self, min_depth: float = 0.001, 
                       min_duration: float = 0.01,
                       snr_threshold: float = 5.0) -> List[Dict[str, float]]:
        """
        Identifica automaticamente eventos de trânsito na curva de luz.
        
        Utiliza algoritmos de detecção de picos invertidos para encontrar
        quedas significativas no fluxo estelar que possam indicar trânsitos
        planetários ou eclipses.
        
        Args:
            min_depth (float): Profundidade mínima do trânsito como fração
                              do fluxo normalizado. Padrão é 0.001 (0.1%).
            min_duration (float): Duração mínima do trânsito em dias.
                                 Padrão é 0.01.
            snr_threshold (float): Limiar de relação sinal-ruído para detecção.
                                  Padrão é 5.0.
                                  
        Returns:
            list: Lista de dicionários com informações dos trânsitos detectados.
                  Cada dicionário contém:
                  - 'time_center': Tempo central do trânsito (BTJD)
                  - 'depth': Profundidade do trânsito
                  - 'duration': Duração estimada em dias
                  - 'snr': Relação sinal-ruído da detecção
                  - 'index': Índice do ponto de mínimo
                  
        Raises:
            RuntimeError: Se nenhuma curva de luz foi carregada.
            
        Example:
            >>> transits = seeker.detect_transits(min_depth=0.005)
            >>> for t in transits:
            ...     print(f"Trânsito em BTJD {t['time_center']:.2f}, "
            ...           f"profundidade: {t['depth']:.4f}")
        """
        if self.lightcurve is None:
            raise RuntimeError("Nenhuma curva de luz carregada.")
        
        logger.info(f"Detectando trânsitos (min_depth={min_depth}, min_duration={min_duration})")
        
        time = self.lightcurve.time.value
        flux = self.lightcurve.flux.value
        
        # Normalizar fluxo
        flux_median = np.median(flux)
        flux_normalized = flux / flux_median
        
        # Calcular ruído
        flux_std = np.std(flux_normalized)
        
        # Inverter sinal para usar detecção de picos
        flux_inverted = -flux_normalized
        
        # Detectar picos (que são mínimos no fluxo original)
        # Usar prominance baseada no ruído
        prominence = snr_threshold * flux_std
        
        peaks, properties = signal.find_peaks(
            flux_inverted,
            prominence=prominence,
            distance=int(min_duration / np.median(np.diff(time)))
        )
        
        transits = []
        for peak_idx in peaks:
            # Calcular profundidade
            depth = 1 - flux_normalized[peak_idx]
            
            if depth < min_depth:
                continue
            
            # Estimar duração usando half-prominence width
            widths = properties['widths']
            peak_idx_in_widths = np.where(properties['peaks'] == peak_idx)[0]
            
            if len(peak_idx_in_widths) > 0:
                width_idx = peak_idx_in_widths[0]
                duration_days = widths[width_idx] * np.median(np.diff(time))
            else:
                duration_days = min_duration
            
            if duration_days < min_duration:
                continue
            
            # Calcular SNR
            snr = depth / flux_std
            
            transit_info = {
                'time_center': float(time[peak_idx]),
                'depth': float(depth),
                'duration': float(duration_days),
                'snr': float(snr),
                'index': int(peak_idx)
            }
            
            transits.append(transit_info)
        
        # Ordenar por SNR decrescente
        transits.sort(key=lambda x: x['snr'], reverse=True)
        
        self.transits = transits
        logger.info(f"Detectados {len(transits)} trânsitos candidatos")
        
        return transits
    
    def export_results(self, output_dir: str = './results',
                      formats: List[str] = ['png', 'csv', 'fits']) -> Dict[str, str]:
        """
        Salva os resultados da análise em múltiplos formatos.
        
        Exporta gráficos, tabelas de dados e arquivos FITS profissionais
        com metadados completos para compartilhamento e análise posterior.
        
        Args:
            output_dir (str): Diretório para salvar os resultados.
            formats (list): Lista de formatos para exportar.
                           Opções: 'png', 'csv', 'fits', 'json'.
                           
        Returns:
            dict: Dicionário mapeando formatos para caminhos dos arquivos salvos.
            
        Raises:
            RuntimeError: Se nenhuma curva de luz foi carregada.
            
        Example:
            >>> results = seeker.export_results(
            ...     output_dir='./analysis_results',
            ...     formats=['png', 'csv', 'fits']
            ... )
            >>> print(f"Arquivos salvos: {results}")
        """
        if self.lightcurve is None:
            raise RuntimeError("Nenhuma curva de luz carregada.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"TIC{self.tic_id}_{timestamp}"
        
        saved_files = {}
        
        # Exportar PNG
        if 'png' in formats:
            try:
                png_path = os.path.join(output_dir, f"{base_filename}_lightcurve.png")
                self.plot_lightcurve(save_path=png_path)
                plt.close()
                
                if self.periodogram is not None:
                    pg_path = os.path.join(output_dir, f"{base_filename}_periodogram.png")
                    self._plot_periodogram()
                    plt.savefig(pg_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    saved_files['periodogram_png'] = pg_path
                
                saved_files['lightcurve_png'] = png_path
                logger.info(f"Gráficos PNG salvos em {output_dir}")
            except Exception as e:
                logger.error(f"Erro ao salvar PNG: {str(e)}")
        
        # Exportar CSV
        if 'csv' in formats:
            try:
                csv_path = os.path.join(output_dir, f"{base_filename}_data.csv")
                
                with open(csv_path, 'w') as f:
                    # Cabeçalho com metadados
                    f.write(f"# TIC: {self.tic_id}\n")
                    f.write(f"# Setor: {self.sector}\n")
                    f.write(f"# Período: {self.primary_period} dias\n")
                    f.write(f"# Trânsitos detectados: {len(self.transits)}\n")
                    f.write(f"# Timestamp: {timestamp}\n")
                    f.write("#\n")
                    f.write("BTJD,Flux,Flux_Err,Quality\n")
                    
                    # Dados
                    for i in range(len(self.lightcurve)):
                        f.write(f"{self.lightcurve.time.value[i]:.6f},"
                               f"{self.lightcurve.flux.value[i]:.6f},"
                               f"{self.lightcurve.flux_err.value[i] if hasattr(self.lightcurve.flux_err, 'value') else 0:.6f},"
                               f"{self.lightcurve.quality[i]}\n")
                
                # Salvar trânsitos em CSV separado
                if self.transits:
                    transits_csv = os.path.join(output_dir, f"{base_filename}_transits.csv")
                    with open(transits_csv, 'w') as f:
                        f.write("Time_Center,BTJD,Depth,Duration_Days,SNR\n")
                        for t in self.transits:
                            f.write(f"{t['time_center']:.6f},{t['depth']:.6f},"
                                   f"{t['duration']:.6f},{t['snr']:.2f}\n")
                    saved_files['transits_csv'] = transits_csv
                
                saved_files['data_csv'] = csv_path
                logger.info(f"Dados CSV salvos em {output_dir}")
            except Exception as e:
                logger.error(f"Erro ao salvar CSV: {str(e)}")
        
        # Exportar FITS
        if 'fits' in formats:
            try:
                fits_path = os.path.join(output_dir, f"{base_filename}.fits")
                
                # Adicionar metadados ao header
                self.lightcurve.meta['TIC_ID'] = self.tic_id
                self.lightcurve.meta['SECTOR'] = self.sector
                self.lightcurve.meta['PERIOD'] = self.primary_period
                self.lightcurve.meta['N_TRANSITS'] = len(self.transits)
                self.lightcurve.meta['ANALYSIS_DATE'] = timestamp
                
                self.lightcurve.to_fits(fits_path, overwrite=True)
                saved_files['fits'] = fits_path
                logger.info(f"Arquivo FITS salvo em {fits_path}")
            except Exception as e:
                logger.error(f"Erro ao salvar FITS: {str(e)}")
        
        # Exportar JSON com resumo
        if 'json' in formats:
            try:
                import json
                json_path = os.path.join(output_dir, f"{base_filename}_summary.json")
                
                summary = {
                    'tic_id': self.tic_id,
                    'sector': self.sector,
                    'author': self.author,
                    'analysis_timestamp': timestamp,
                    'primary_period_days': self.primary_period,
                    'n_data_points': len(self.lightcurve),
                    'time_range': {
                        'start': float(self.lightcurve.time.value[0]),
                        'end': float(self.lightcurve.time.value[-1])
                    },
                    'transits_detected': len(self.transits),
                    'transits': self.transits
                }
                
                with open(json_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                saved_files['summary_json'] = json_path
                logger.info(f"Resumo JSON salvo em {json_path}")
            except Exception as e:
                logger.error(f"Erro ao salvar JSON: {str(e)}")
        
        return saved_files
    
    def interactive_mode(self) -> None:
        """
        Inicia uma sessão interativa guiando o usuário passo a passo.
        
        Este método fornece uma interface de linha de comando interativa
        que guia astrônomos amadores através do processo completo de análise,
        desde a busca por TIC até a exportação de resultados.
        
        O modo interativo inclui:
        - Explicações conceituais
        - Validação de entrada do usuário
        - Tratamento robusto de erros
        - Sugestões automáticas
        
        Example:
            >>> seeker = StellarSeeker(tic_id=261136679)
            >>> seeker.interactive_mode()
        """
        print("\n" + "="*70)
        print("  STELLARSEEKER - Análise de Curvas de Luz e Exoplanetas")
        print("="*70)
        print("\nBem-vindo ao StellarSeeker!")
        print("Esta ferramenta permite analisar dados do telescópio espacial TESS")
        print("para descobrir exoplanetas e estudar estrelas variáveis.\n")
        
        while True:
            print("\n" + "-"*50)
            print("MENU PRINCIPAL:")
            print("1. Buscar setores disponíveis")
            print("2. Baixar curva de luz")
            print("3. Visualizar curva de luz")
            print("4. Calcular periodograma")
            print("5. Obter período candidato")
            print("6. Dobrar fase (phase folding)")
            print("7. Detectar trânsitos")
            print("8. Exportar resultados")
            print("9. Analisar exemplo demonstrativo")
            print("0. Sair")
            print("-"*50)
            
            choice = input("\nEscolha uma opção (0-9): ").strip()
            
            try:
                if choice == '1':
                    self._interactive_search()
                elif choice == '2':
                    self._interactive_download()
                elif choice == '3':
                    self._interactive_plot()
                elif choice == '4':
                    self._interactive_periodogram()
                elif choice == '5':
                    self._interactive_get_period()
                elif choice == '6':
                    self._interactive_fold()
                elif choice == '7':
                    self._interactive_detect()
                elif choice == '8':
                    self._interactive_export()
                elif choice == '9':
                    self._interactive_example()
                elif choice == '0':
                    print("\nObrigado por usar StellarSeeker! Boas descobertas! 🌟\n")
                    break
                else:
                    print("Opção inválida. Escolha entre 0 e 9.")
                    
            except KeyboardInterrupt:
                print("\n\nOperação cancelada pelo usuário.")
            except Exception as e:
                print(f"\n❌ Erro: {str(e)}")
                print("Dica: Verifique sua conexão com a internet e tente novamente.")
    
    def _interactive_search(self):
        """Sub-rotina interativa para busca de setores."""
        print("\n🔍 Buscando setores disponíveis...")
        sectors = self.search_sectors()
        
        if not sectors:
            print("⚠️ Nenhum setor encontrado para este TIC.")
            return
        
        print(f"\n✅ Encontrados {len(sectors)} setores:\n")
        for s in sectors:
            print(f"  [{s['index']}] Setor {s['sector']}: {s['duration']:.1f} dias")
        
        if sectors:
            selected = input(f"\nSelecione um setor (0-{len(sectors)-1}): ").strip()
            try:
                self.sector = sectors[int(selected)]['sector']
                print(f"Setor {self.sector} selecionado.")
            except (ValueError, IndexError):
                print("Seleção inválida. Use o primeiro setor por padrão.")
                self.sector = sectors[0]['sector']
    
    def _interactive_download(self):
        """Sub-rotina interativa para download."""
        if self._search_results is None:
            print("⚠️ Primeiro busque os setores disponíveis (opção 1).")
            return
        
        print("\n📥 Baixando dados da curva de luz...")
        try:
            lc = self.download_lightcurve()
            print(f"✅ Download concluído: {len(lc)} pontos de dados")
        except Exception as e:
            print(f"❌ Erro no download: {str(e)}")
    
    def _interactive_plot(self):
        """Sub-rotina interativa para plotagem."""
        if self.lightcurve is None:
            print("⚠️ Primeiro baixe a curva de luz (opção 2).")
            return
        
        print("\n📊 Gerando gráfico da curva de luz...")
        normalize = input("Normalizar fluxo? (s/n, padrão=s): ").strip().lower() != 'n'
        
        try:
            fig = self.plot_lightcurve(normalize=normalize)
            plt.show()
            print("✅ Gráfico gerado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao gerar gráfico: {str(e)}")
    
    def _interactive_periodogram(self):
        """Sub-rotina interativa para periodograma."""
        if self.lightcurve is None:
            print("⚠️ Primeiro baixe a curva de luz (opção 2).")
            return
        
        print("\n📈 Calculando periodograma...")
        try:
            pg_min = float(input("Período mínimo (dias, padrão=0.1): ") or "0.1")
            pg_max = float(input("Período máximo (dias, padrão=30): ") or "30")
            
            self.calculate_periodogram(
                minimum_period=pg_min,
                maximum_period=pg_max,
                plot=True
            )
            plt.show()
            print("✅ Periodograma calculado!")
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
    
    def _interactive_get_period(self):
        """Sub-rotina interativa para obter período."""
        if self.periodogram is None:
            print("⚠️ Primeiro calcule o periodograma (opção 4).")
            return
        
        period = self.get_primary_period()
        print(f"\n🎯 Período candidato: {period:.6f} dias")
        print(f"   ({format_period(period)})")
    
    def _interactive_fold(self):
        """Sub-rotina interativa para dobradura de fase."""
        if self.lightcurve is None:
            print("⚠️ Primeiro baixe a curva de luz (opção 2).")
            return
        
        print("\n🔄 Dobrando fase...")
        use_calculated = input("Usar período calculado? (s/n, padrão=s): ").strip().lower()
        
        period = None
        if use_calculated != 's':
            try:
                period = float(input("Informe o período (dias): "))
            except ValueError:
                print("Período inválido. Usando período calculado.")
        
        try:
            fig, _, _ = self.fold_phase(period=period)
            plt.show()
            print("✅ Fase dobrada com sucesso!")
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
    
    def _interactive_detect(self):
        """Sub-rotina interativa para detecção de trânsitos."""
        if self.lightcurve is None:
            print("⚠️ Primeiro baixe a curva de luz (opção 2).")
            return
        
        print("\n🔎 Detectando trânsitos...")
        try:
            min_depth = float(input("Profundidade mínima (padrão=0.001): ") or "0.001")
            transits = self.detect_transits(min_depth=min_depth)
            
            if transits:
                print(f"\n✅ Detectados {len(transits)} trânsitos candidatos:\n")
                for i, t in enumerate(transits[:5], 1):
                    print(f"  {i}. BTJD {t['time_center']:.2f}, "
                         f"profundidade: {t['depth']:.4f}, "
                         f"SNR: {t['snr']:.1f}")
                if len(transits) > 5:
                    print(f"  ... e mais {len(transits) - 5} trânsitos")
            else:
                print("⚠️ Nenhum trânsito detectado com esses parâmetros.")
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
    
    def _interactive_export(self):
        """Sub-rotina interativa para exportação."""
        if self.lightcurve is None:
            print("⚠️ Primeiro baixe a curva de luz (opção 2).")
            return
        
        print("\n💾 Exportando resultados...")
        output_dir = input("Diretório de saída (padrão=./results): ").strip() or "./results"
        
        try:
            results = self.export_results(output_dir=output_dir)
            print(f"\n✅ Arquivos salvos:\n")
            for fmt, path in results.items():
                print(f"  - {path}")
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
    
    def _interactive_example(self):
        """Sub-rotina interativa para exemplos demonstrativos."""
        print("\n📚 Exemplos demonstrativos:")
        print("1. π Mensae (TIC 261136679) - Exoplaneta confirmado (~6.26 dias)")
        print("2. Algol/Beta Persei (TIC 346783960) - Binária eclipsante (~2.87 dias)")
        print("3. TYC 7037-89-1 (TIC 168789840) - Sistema sextuplo (~1.57 dias)")
        
        choice = input("\nEscolha um exemplo (1-3): ").strip()
        
        examples = {
            '1': 261136679,
            '2': 346783960,
            '3': 168789840
        }
        
        if choice in examples:
            new_tic = examples[choice]
            print(f"\n🔄 Reiniciando com TIC {new_tic}...")
            # Nota: Em uma implementação real, isso criaria uma nova instância
            print("Reinicie o StellarSeeker com o novo TIC para análise completa.")
        else:
            print("Escolha inválida.")
