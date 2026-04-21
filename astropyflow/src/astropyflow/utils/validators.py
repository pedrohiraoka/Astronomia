"""
Validadores para verificação de qualidade de dados astronômicos.

Implementa verificações para:
- Valores nulos/NaN
- Outliers estatísticos
- Unidades inconsistentes
- Faixas válidas para grandezas astronômicas
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from astropy import units as u


logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validador de qualidade de dados astronômicos.

    Atributos:
        tolerance_sigma: Número de desvios padrão para detecção de outliers.
        min_values: Quantidade mínima de valores válidos para aprovação.

    Example:
        >>> validator = DataValidator(tolerance_sigma=3.0)
        >>> result = validator.check_nulls(df)
        >>> if result['has_issues']:
        ...     logger.warning("Dados com problemas encontrados")
    """

    def __init__(self, tolerance_sigma: float = 3.0, min_valid_ratio: float = 0.95):
        """
        Inicializa o validador.

        Args:
            tolerance_sigma: Limite em desvios padrão para outliers (default: 3.0).
            min_valid_ratio: Proporção mínima de dados válidos (default: 0.95).
        """
        self.tolerance_sigma = tolerance_sigma
        self.min_valid_ratio = min_valid_ratio
        self.validation_log: List[Dict[str, Any]] = []

    def check_nulls(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Verifica presença de valores nulos ou NaN no DataFrame.

        Args:
            data: DataFrame para validação.

        Returns:
            Dicionário com:
                - has_issues: Booleano indicando se há problemas.
                - null_counts: Contagem de nulos por coluna.
                - null_ratio: Proporção de nulos por coluna.
                - recommendation: Sugestão de ação.
        """
        null_counts = data.isnull().sum()
        total_rows = len(data)
        null_ratio = null_counts / total_rows if total_rows > 0 else pd.Series(dtype=float)

        has_issues = (null_ratio > (1 - self.min_valid_ratio)).any()

        result = {
            "has_issues": has_issues,
            "null_counts": null_counts.to_dict(),
            "null_ratio": null_ratio.to_dict(),
            "recommendation": "",
        }

        if has_issues:
            cols_with_nulls = null_ratio[null_ratio > 0].index.tolist()
            result["recommendation"] = (
                f"Colunas com nulos: {cols_with_nulls}. "
                "Considere imputação ou remoção de linhas afetadas."
            )
            logger.warning(
                f"Valores nulos detectados em {len(cols_with_nulls)} colunas. "
                f"Máximo ratio: {null_ratio.max():.2%}"
            )
        else:
            result["recommendation"] = "Dados OK - sem valores nulos significativos."

        self.validation_log.append({"check": "nulls", "result": result})
        return result

    def check_outliers(
        self, data: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detecta outliers usando método Z-score.

        Args:
            data: DataFrame para validação.
            columns: Lista de colunas para verificar. Se None, usa todas numéricas.

        Returns:
            Dicionário com:
                - has_issues: Booleano indicando se há outliers.
                - outlier_counts: Contagem de outliers por coluna.
                - outlier_indices: Índices das linhas com outliers.
                - recommendation: Sugestão de ação.
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info: Dict[str, Any] = {
            "has_issues": False,
            "outlier_counts": {},
            "outlier_indices": set(),
            "details": {},
        }

        for col in columns:
            if col not in data.columns:
                continue

            series = data[col].dropna()
            if len(series) < 3:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0 or np.isnan(std):
                continue

            z_scores = np.abs((series - mean) / std)
            outliers = z_scores > self.tolerance_sigma

            n_outliers = outliers.sum()
            if n_outliers > 0:
                outlier_info["has_issues"] = True
                outlier_info["outlier_counts"][col] = int(n_outliers)
                outlier_info["outlier_indices"].update(
                    series[outliers].index.tolist()
                )
                outlier_info["details"][col] = {
                    "mean": float(mean),
                    "std": float(std),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "outlier_values": series[outliers].tolist()[:10],  # Primeiros 10
                }

        outlier_info["outlier_indices"] = list(outlier_info["outlier_indices"])

        if outlier_info["has_issues"]:
            total_outliers = sum(outlier_info["outlier_counts"].values())
            outlier_info["recommendation"] = (
                f"{total_outliers} outliers detectados em {len(outlier_info['outlier_counts'])} colunas. "
                "Revise dados extremos e considere tratamento específico."
            )
            logger.warning(
                f"Outliers detectados: {total_outliers} valores além de "
                f"{self.tolerance_sigma}σ da média."
            )
        else:
            outlier_info["recommendation"] = "Sem outliers significativos detectados."

        self.validation_log.append({"check": "outliers", "result": outlier_info})
        return outlier_info

    def check_units(
        self, data: pd.DataFrame, expected_units: Dict[str, u.UnitBase]
    ) -> Dict[str, Any]:
        """
        Verifica consistência de unidades em colunas do DataFrame.

        Args:
            data: DataFrame com dados astronômicos.
            expected_units: Dicionário mapeando colunas para unidades esperadas.
                           Ex: {"ra": u.deg, "distance": u.pc}

        Returns:
            Dicionário com:
                - has_issues: Booleano indicando problemas de unidade.
                - unit_status: Status por coluna.
                - recommendation: Sugestão de ação.
        """
        unit_status: Dict[str, str] = {}
        has_issues = False

        for col, expected_unit in expected_units.items():
            if col not in data.columns:
                unit_status[col] = "coluna_nao_encontrada"
                has_issues = True
                logger.warning(f"Coluna '{col}' não encontrada para verificação de unidades.")
                continue

            # Tenta converter valores para a unidade esperada
            try:
                values = data[col].dropna().values
                if len(values) == 0:
                    unit_status[col] = "sem_dados"
                    continue

                # Assume que valores estão na unidade correta se conversão funcionar
                test_quantity = values[0] * expected_unit
                unit_status[col] = "ok"

            except (ValueError, TypeError) as e:
                unit_status[col] = f"erro_conversao: {str(e)}"
                has_issues = True
                logger.error(f"Erro ao verificar unidades em '{col}': {e}")

        result = {
            "has_issues": has_issues,
            "unit_status": unit_status,
            "recommendation": "",
        }

        if has_issues:
            result["recommendation"] = (
                "Problemas de unidades detectados. Verifique conversões necessárias."
            )
        else:
            result["recommendation"] = "Unidades consistentes."

        self.validation_log.append({"check": "units", "result": result})
        return result

    def check_ranges(
        self, data: pd.DataFrame, valid_ranges: Dict[str, Tuple[Optional[float], Optional[float]]]
    ) -> Dict[str, Any]:
        """
        Verifica se valores estão dentro de faixas astronômicas válidas.

        Args:
            data: DataFrame com dados.
            valid_ranges: Dicionário mapeando colunas para tuplas (min, max).
                         Use None para limites abertos. Ex: {"ra": (0, 360)}

        Returns:
            Dicionário com status das verificações de faixa.
        """
        range_status: Dict[str, Any] = {}
        has_issues = False

        for col, (min_val, max_val) in valid_ranges.items():
            if col not in data.columns:
                continue

            series = data[col].dropna()
            if len(series) == 0:
                continue

            out_of_range = pd.Series(False, index=data.index)

            if min_val is not None:
                out_of_range |= data[col] < min_val
            if max_val is not None:
                out_of_range |= data[col] > max_val

            n_out_of_range = out_of_range.sum()
            if n_out_of_range > 0:
                has_issues = True
                range_status[col] = {
                    "expected": f"[{min_val}, {max_val}]",
                    "actual_min": float(series.min()),
                    "actual_max": float(series.max()),
                    "out_of_range_count": int(n_out_of_range),
                }
                logger.warning(
                    f"Coluna '{col}': {n_out_of_range} valores fora da faixa "
                    f"[{min_val}, {max_val}]"
                )
            else:
                range_status[col] = {"status": "ok"}

        result = {
            "has_issues": has_issues,
            "range_status": range_status,
            "recommendation": "",
        }

        if has_issues:
            result["recommendation"] = (
                "Valores fora de faixas esperadas detectados. Revise dados."
            )
        else:
            result["recommendation"] = "Todos os valores dentro de faixas válidas."

        self.validation_log.append({"check": "ranges", "result": result})
        return result

    def validate_full(
        self,
        data: pd.DataFrame,
        expected_units: Optional[Dict[str, u.UnitBase]] = None,
        valid_ranges: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Executa todas as validações em sequência.

        Args:
            data: DataFrame para validação completa.
            expected_units: Unidades esperadas por coluna (opcional).
            valid_ranges: Faixas válidas por coluna (opcional).

        Returns:
            Dicionário consolidado com resultados de todas as validações.
        """
        self.validation_log = []

        results = {
            "nulls": self.check_nulls(data),
            "outliers": self.check_outliers(data),
        }

        if expected_units:
            results["units"] = self.check_units(data, expected_units)

        if valid_ranges:
            results["ranges"] = self.check_ranges(data, valid_ranges)

        # Resumo geral
        has_any_issue = any(r.get("has_issues", False) for r in results.values())
        results["summary"] = {
            "is_valid": not has_any_issue,
            "total_checks": len(results),
            "checks_passed": sum(1 for r in results.values() if not r.get("has_issues", False)),
        }

        if has_any_issue:
            logger.warning("Validação completa: DADOS COM PROBLEMAS DETECTADOS")
        else:
            logger.info("Validação completa: DADOS APROVADOS")

        return results
