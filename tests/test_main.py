import os
import tempfile

import pandas as pd

from main import SCORE_PATTERN, load_existing_records, save_records, sha256_hash


class TestSHA256Hash:
    def test_sha256_hash_deterministic(self):
        assert sha256_hash("hello") == sha256_hash("hello")

    def test_sha256_hash_different_inputs(self):
        assert sha256_hash("hello") != sha256_hash("world")


class TestScorePattern:
    def test_score_pattern_valid(self):
        match = SCORE_PATTERN.search("85/100")
        assert match is not None
        assert match.group(1) == "85"

    def test_score_pattern_no_match(self):
        assert SCORE_PATTERN.search("no score") is None


class TestLoadExistingRecords:
    def test_load_existing_records_missing_file(self):
        result = load_existing_records("/nonexistent/path.csv")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_load_existing_records_corrupted(self, caplog):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("not a valid csv")
            tmpfile = f.name
        try:
            result = load_existing_records(tmpfile)
            assert isinstance(result, pd.DataFrame)
            assert result.empty
            assert "starting fresh" in caplog.text
        finally:
            os.unlink(tmpfile)

    def test_load_existing_records_valid(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("key,run,score\nabc,0,85\ndef,1,90")
            tmpfile = f.name
        try:
            result = load_existing_records(tmpfile)
            assert isinstance(result, pd.DataFrame)
            assert "run" in result.columns
            assert "score" in result.columns
            assert result.index.name == "key"
            assert len(result) == 2
        finally:
            os.unlink(tmpfile)

    def test_save_and_load_roundtrip(self):
        df = pd.DataFrame({"key": ["k1", "k2"], "run": [0, 1], "score": [85, 90]})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            tmpfile = f.name
        try:
            save_records(df, tmpfile)
            loaded = load_existing_records(tmpfile)
            pd.testing.assert_frame_equal(df, loaded)
        finally:
            os.unlink(tmpfile)
