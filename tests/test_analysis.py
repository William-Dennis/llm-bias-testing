"""Tests for slm_bias_testing.analysis — statistical helpers."""

from __future__ import annotations

import pandas as pd

from slm_bias_testing.analysis import (
    build_summary_table,
    cohens_d,
    group_summary,
    pairwise_comparisons,
    per_cv_variance,
    variance_breakdown,
)


class TestCohensD:
    def test_identical_groups(self):
        s1 = pd.Series([10, 10, 10])
        s2 = pd.Series([10, 10, 10])
        assert cohens_d(s1, s2) == 0.0

    def test_different_groups(self):
        s1 = pd.Series([10, 12, 14])
        s2 = pd.Series([20, 22, 24])
        d = cohens_d(s1, s2)
        assert d < 0  # s1 < s2

    def test_small_sample(self):
        s1 = pd.Series([10])
        s2 = pd.Series([20])
        assert cohens_d(s1, s2) == 0.0  # n < 2

    def test_zero_pooled_std(self):
        s1 = pd.Series([5, 5])
        s2 = pd.Series([5, 5])
        assert cohens_d(s1, s2) == 0.0


class TestGroupSummary:
    def test_basic(self, sample_df):
        result = group_summary(sample_df, "name")
        assert not result.empty
        assert "mean" in result.columns
        assert "count" in result.columns

    def test_missing_column(self, sample_df):
        result = group_summary(sample_df, "nonexistent")
        assert result.empty

    def test_all_nan_column(self):
        df = pd.DataFrame({"group": [None, None], "score": [1, 2]})
        result = group_summary(df, "group")
        assert result.empty


class TestPairwiseComparisons:
    def test_basic(self, sample_df):
        result = pairwise_comparisons(sample_df, "name")
        assert not result.empty
        assert "cohens_d" in result.columns
        assert "p_value" in result.columns

    def test_single_group(self):
        df = pd.DataFrame({"group": ["A", "A"], "score": [1, 2]})
        result = pairwise_comparisons(df, "group")
        assert result.empty

    def test_missing_column(self, sample_df):
        result = pairwise_comparisons(sample_df, "nonexistent")
        assert result.empty


class TestVarianceBreakdown:
    def test_basic(self, sample_df):
        result = variance_breakdown(sample_df, ["name", "university"])
        assert "name" in result
        assert "proportion" in result["name"]

    def test_zero_total_variance(self):
        df = pd.DataFrame({"factor": ["A", "B"], "score": [5.0, 5.0]})
        result = variance_breakdown(df, ["factor"])
        assert result == {}

    def test_missing_column(self, sample_df):
        result = variance_breakdown(sample_df, ["nonexistent"])
        assert "nonexistent" not in result


class TestPerCvVariance:
    def test_basic(self, sample_df):
        cv_std, summary = per_cv_variance(sample_df, "key", "score")
        assert len(cv_std) > 0
        assert "mean_cv_std" in summary

    def test_missing_key_column(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        cv_std, summary = per_cv_variance(df)
        assert cv_std.empty
        assert summary == {}


class TestBuildSummaryTable:
    def test_returns_string(self, sample_df):
        result = build_summary_table(sample_df, ["name", "university"])
        assert isinstance(result, str)
        assert "STATISTICAL ANALYSIS" in result

    def test_empty_dataframe(self):
        df = pd.DataFrame({"score": pd.Series(dtype=float)})
        result = build_summary_table(df, [])
        assert "STATISTICAL ANALYSIS" in result
