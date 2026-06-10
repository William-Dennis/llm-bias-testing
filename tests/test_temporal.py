"""Tests for slm_bias_testing.temporal — result scanning and merging."""

from __future__ import annotations

import json
import os
import tempfile

from slm_bias_testing.temporal import find_results, merge_registry


class TestFindResults:
    def test_finds_results_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "model-a", "stereoset")
            os.makedirs(subdir)
            data = {
                "model": "smollm-135m",
                "benchmark": "stereoset",
                "overall_stereotype_score": 42.0,
            }
            with open(os.path.join(subdir, "results.json"), "w") as f:
                json.dump(data, f)

            records = find_results(tmpdir)
            assert len(records) == 1
            assert records[0]["model"] == "smollm-135m"
            assert records[0]["score"] == 42.0

    def test_ignores_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "model-a", "stereoset")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "results.json"), "w") as f:
                f.write("not json{")

            records = find_results(tmpdir)
            assert len(records) == 0

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = find_results(tmpdir)
            assert len(records) == 0

    def test_skips_unknown_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "unknown-model", "stereoset")
            os.makedirs(subdir)
            data = {"model": "unknown-model", "benchmark": "stereoset"}
            with open(os.path.join(subdir, "results.json"), "w") as f:
                json.dump(data, f)

            records = find_results(tmpdir)
            assert len(records) == 1  # find_results doesn't filter by registry


class TestMergeRegistry:
    def test_merges_known_model(self):
        records = [
            {"model": "smollm-135m", "benchmark": "stereoset", "score": 42.0, "n_examples": 100}
        ]
        df = merge_registry(records)
        assert len(df) == 1
        assert df.iloc[0]["family"] == "huggingface"
        assert df.iloc[0]["params"] == 135_000_000

    def test_skips_unknown_model(self):
        records = [
            {"model": "nonexistent", "benchmark": "stereoset", "score": 42.0, "n_examples": 100}
        ]
        df = merge_registry(records)
        assert df.empty

    def test_empty_records(self):
        df = merge_registry([])
        assert df.empty
