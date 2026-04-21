"""
Testes para o módulo data_handler.
"""

import numpy as np
import pandas as pd
import pytest

from astropyflow.core.data_handler import (
    create_mock_catalog,
    validate_catalog,
)
from astropyflow.utils.validators import DataValidator


class TestDataValidator:
    """Testes para a classe DataValidator."""

    def test_check_nulls_no_nulls(self):
        """Testa verificação de nulos quando não há nulos."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        validator = DataValidator()
        result = validator.check_nulls(df)

        assert result["has_issues"] == False
        assert all(v == 0 for v in result["null_counts"].values())

    def test_check_nulls_with_nulls(self):
        """Testa verificação de nulos com dados faltantes."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4.0, 5.0, np.nan]})
        # Com 2/3=67% válidos e min_valid_ratio=0.5, não deve ter issues
        # Para ter issues, precisamos de menos de 50% válidos
        validator = DataValidator(min_valid_ratio=0.8)  # Precisa de 80% válidos
        result = validator.check_nulls(df)

        assert result["has_issues"] == True
        assert result["null_counts"]["a"] == 1
        assert result["null_counts"]["b"] == 1

    def test_check_outliers_detects(self):
        """Testa detecção de outliers."""
        np.random.seed(42)
        data = np.random.randn(100) * 10 + 50
        data[0] = 500  # Outlier extremo
        df = pd.DataFrame({"values": data})

        validator = DataValidator(tolerance_sigma=3.0)
        result = validator.check_outliers(df)

        assert result["has_issues"] is True
        assert result["outlier_counts"]["values"] >= 1

    def test_validate_full(self):
        """Testa validação completa."""
        df = pd.DataFrame({
            "ra": np.random.uniform(0, 360, 50),
            "dec": np.random.uniform(-90, 90, 50),
            "value": np.random.randn(50) * 10,
        })

        validator = DataValidator()
        result = validator.validate_full(
            df,
            valid_ranges={"ra": (0, 360), "dec": (-90, 90)},
        )

        assert "summary" in result
        assert "is_valid" in result["summary"]


class TestCreateMockCatalog:
    """Testes para criação de catálogo mock."""

    def test_create_catalog_size(self):
        """Testa se catálogo tem tamanho esperado."""
        catalog = create_mock_catalog(n_stars=100, seed=42)
        assert len(catalog) == 100
        assert len(catalog.columns) > 5

    def test_create_catalog_columns(self):
        """Testa se colunas esperadas estão presentes."""
        catalog = create_mock_catalog(n_stars=50, seed=42)
        expected_cols = ["star_id", "ra", "dec", "distance_pc", "apparent_mag",
                         "absolute_mag", "spectral_class", "temperature_K"]

        for col in expected_cols:
            assert col in catalog.columns

    def test_create_catalog_ranges(self):
        """Testa se valores estão em faixas astronômicas válidas."""
        catalog = create_mock_catalog(n_stars=100, seed=42)

        assert catalog["ra"].min() >= 0
        assert catalog["ra"].max() <= 360
        assert catalog["dec"].min() >= -90
        assert catalog["dec"].max() <= 90
        assert catalog["distance_pc"].min() > 0
        assert catalog["temperature_K"].min() > 1000

    def test_create_catalog_reproducibility(self):
        """Testa reprodutibilidade com seed."""
        cat1 = create_mock_catalog(n_stars=50, seed=123)
        cat2 = create_mock_catalog(n_stars=50, seed=123)

        pd.testing.assert_frame_equal(cat1, cat2)


class TestValidateCatalog:
    """Testes para validação de catálogos."""

    def test_validate_clean_catalog(self):
        """Testa validação de catálogo limpo."""
        catalog = create_mock_catalog(n_stars=100, seed=42)
        result = validate_catalog(catalog)

        assert "is_valid" in result
        assert "cleaned_data" in result
        assert result["cleaned_rows"] > 0

    def test_validate_removes_nulls(self):
        """Testa que validação remove linhas com nulos críticos."""
        catalog = create_mock_catalog(n_stars=100, seed=42)
        # Adicionar nulos em colunas críticas
        catalog.loc[0:4, "ra"] = np.nan

        result = validate_catalog(catalog)

        assert result["rows_removed"] >= 5
        assert result["cleaned_data"]["ra"].isnull().sum() == 0

    def test_validate_returns_summary(self):
        """Testa se resumo da validação é retornado."""
        catalog = create_mock_catalog(n_stars=50, seed=42)
        result = validate_catalog(catalog)

        assert "original_rows" in result
        assert "cleaned_rows" in result
        assert "validation_details" in result
