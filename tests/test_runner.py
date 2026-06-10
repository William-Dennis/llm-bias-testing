from unittest.mock import patch

from slm_bias_testing.runner import run_benchmark_for_model


class TestRunBenchmarkForModel:
    @patch("slm_bias_testing.runner.pull_model", return_value=True)
    @patch("slm_bias_testing.runner.get_model")
    def test_skip_existing_results(self, mock_get_model, mock_pull):
        import json
        import os
        import tempfile

        mock_get_model.return_value = {"ollama_tag": "smollm:135m"}

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "smollm-135m", "cv-screening")
            os.makedirs(results_dir)
            results_file = os.path.join(results_dir, "results.json")
            with open(results_file, "w") as f:
                json.dump({"model": "smollm-135m"}, f)

            run_benchmark_for_model("smollm-135m", "cv-screening", tmpdir, 1800)

            # pull_model is called once (before loop), but benchmark is skipped
            mock_pull.assert_called_once()

    @patch("slm_bias_testing.runner.pull_model", return_value=False)
    @patch("slm_bias_testing.runner.get_model")
    def test_skip_on_pull_failure(self, mock_get_model, mock_pull):
        import os
        import tempfile

        mock_get_model.return_value = {"ollama_tag": "smollm:135m"}

        with tempfile.TemporaryDirectory() as tmpdir:
            run_benchmark_for_model("smollm-135m", "cv-screening", tmpdir, 1800)

            # Results dir should not be created
            results_dir = os.path.join(tmpdir, "smollm-135m", "cv-screening")
            assert not os.path.exists(results_dir)
