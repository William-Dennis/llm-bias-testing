import tempfile
from unittest.mock import patch

from slm_bias_testing.benchmarks import BaseBenchmark
from slm_bias_testing.benchmarks.stereoset import StereoSetBenchmark
from slm_bias_testing.benchmarks.winobias import WinoBiasBenchmark
from slm_bias_testing.benchmarks.demographic_bias import DemographicBiasBenchmark


class TestBaseBenchmark:
    def test_interface(self):
        class ConcreteBench(BaseBenchmark):
            name = "test"

            def load_dataset(self):
                return []

            def evaluate(self, model):
                return {"result": 42}

        bm = ConcreteBench()
        assert bm.load_dataset() == []
        assert bm.evaluate(None) == {"result": 42}

    def test_save_and_load_results(self):
        class ConcreteBench(BaseBenchmark):
            name = "test"

            def load_dataset(self):
                return []

            def evaluate(self, model):
                return {"result": 42}

        bm = ConcreteBench()
        with tempfile.TemporaryDirectory() as tmpdir:
            bm.save_results({"result": 42}, tmpdir)
            loaded = bm.load_results(tmpdir)
            assert loaded == {"result": 42}

    def test_load_results_nonexistent(self):
        class ConcreteBench(BaseBenchmark):
            name = "test"

            def load_dataset(self):
                return []

            def evaluate(self, model):
                return {"result": 42}

        bm = ConcreteBench()
        with tempfile.TemporaryDirectory() as tmpdir:
            assert bm.load_results(tmpdir) is None


class MockModel:
    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}

    def predict(self, prompt: str, temperature: float = 0.0) -> str:
        for key, val in self.responses.items():
            if key in prompt:
                return val
        return "50"


class TestStereoSetBenchmark:
    def _make_stereoset_item(self, id, bias_type, target, context, stereotype, anti_stereotype):
        """Helper to create a StereoSet item in the real nested format."""
        return {
            "id": id,
            "bias_type": bias_type,
            "target": target,
            "context": context,
            "sentences": {
                "sentence": [stereotype, "unrelated sentence", anti_stereotype],
                "gold_label": [1, 0, 2],  # 1=stereotype, 0=unrelated, 2=anti-stereotype
            },
        }

    def test_load_dataset(self):
        fake_data = [
            self._make_stereoset_item(
                "test1", "gender", "doctor", "The doctor entered.",
                "He is skilled.", "She is skilled."
            )
        ]

        with patch("slm_bias_testing.benchmarks.stereoset.datasets.load_dataset") as mock_load:
            mock_load.return_value = fake_data
            bm = StereoSetBenchmark()
            data = bm.load_dataset()
            assert len(data) == 1
            assert data[0]["bias_type"] == "gender"
            mock_load.assert_called_once_with("stereoset", "intrasentence", split="validation")

    def test_evaluate_chosen_stereotype(self):
        fake_data = [
            self._make_stereoset_item(
                "test1", "gender", "doctor", "The doctor entered.",
                "He is skilled.", "She is skilled."
            )
        ]

        model = MockModel({"He is skilled.": "90", "She is skilled.": "10"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(model)
            assert results["benchmark"] == "stereoset"
            assert results["n_examples"] == 1
            assert results["overall_stereotype_score"] == 100.0

    def test_evaluate_chosen_anti_stereotype(self):
        fake_data = [
            self._make_stereoset_item(
                "test1", "gender", "doctor", "The doctor entered.",
                "He is skilled.", "She is skilled."
            )
        ]

        model = MockModel({"He is skilled.": "10", "She is skilled.": "90"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(model)
            assert results["overall_stereotype_score"] == 0.0

    def test_per_category(self):
        fake_data = [
            self._make_stereoset_item(
                "test1", "gender", "doctor", "The doctor entered.",
                "He is skilled.", "She is skilled."
            ),
            self._make_stereoset_item(
                "test2", "race", "person", "The person walked in.",
                "He is loud.", "She is quiet."
            ),
        ]

        model = MockModel({"He is skilled.": "90", "She is skilled.": "10", "He is loud.": "80", "She is quiet.": "20"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(model)
            assert results["per_category"]["gender"] == 100.0
            assert results["per_category"]["race"] == 100.0

    def test_max_samples_limits_evaluation(self):
        """Test that max_samples truncates the dataset."""
        fake_data = [
            self._make_stereoset_item(
                "test1", "gender", "doctor", "The doctor entered.",
                "He is skilled.", "She is skilled."
            ),
            self._make_stereoset_item(
                "test2", "race", "person", "The person walked in.",
                "He is loud.", "She is quiet."
            ),
            self._make_stereoset_item(
                "test3", "age", "worker", "The worker arrived.",
                "He is experienced.", "She is experienced."
            ),
        ]

        model = MockModel({"He is skilled.": "90", "She is skilled.": "10"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(model, max_samples=1)
            assert results["n_examples"] == 1

    def test_missing_gold_label_skipped(self):
        """Test that items without stereotype/anti-stereotype are skipped."""
        fake_data = [
            {
                "id": "bad1",
                "bias_type": "gender",
                "target": "doctor",
                "context": "The doctor entered.",
                "sentences": {
                    "sentence": ["Only unrelated here"],
                    "gold_label": [0],
                },
            },
            self._make_stereoset_item(
                "good1", "gender", "doctor", "The doctor entered.",
                "He is skilled.", "She is skilled."
            ),
        ]

        model = MockModel({"He is skilled.": "90", "She is skilled.": "10"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(model)
            assert results["n_examples"] == 1


class TestWinoBiasBenchmark:
    def test_load_dataset(self):
        bm = WinoBiasBenchmark()
        data = bm.load_dataset()
        assert len(data) == 1584
        configs = set(d["config"] for d in data)
        assert configs == {"type1_pro", "type1_anti", "type2_pro", "type2_anti"}

    def test_occupations_set(self):
        bm = WinoBiasBenchmark()
        bm.load_dataset()
        occ = bm._get_occupations()
        assert "developer" in occ
        assert "nurse" in occ
        assert len(occ) == 40

    def test_entity_extraction(self):
        bm = WinoBiasBenchmark()
        data = bm.load_dataset()
        item = [d for d in data if d["config"] == "type1_pro"][0]
        tokens = item["tokens"]
        coref = [int(x) for x in item["coreference_clusters"]]
        entities = bm._find_entities(tokens, coref[2])
        assert len(entities) == 2
        correct = bm._extract_entity_name(tokens, coref[0], coref[1])
        assert correct in entities

    def test_compute_metrics(self):
        bm = WinoBiasBenchmark()
        fake_results = [
            {"correct": True, "is_pro": True, "config": "type1_pro", "pronoun": "he"},
            {"correct": False, "is_pro": False, "config": "type1_anti", "pronoun": "she"},
            {"correct": True, "is_pro": True, "config": "type1_pro", "pronoun": "he"},
            {"correct": True, "is_pro": False, "config": "type1_anti", "pronoun": "she"},
        ]
        metrics = bm._compute_metrics(fake_results)
        assert metrics["overall_accuracy"] == 75.0
        assert metrics["pro_accuracy"] == 100.0
        assert metrics["anti_accuracy"] == 50.0
        assert metrics["bias_score"] == 50.0
        assert metrics["n_examples"] == 4


class TestDemographicBiasBenchmark:
    def test_load_dataset(self):
        bm = DemographicBiasBenchmark()
        data = bm.load_dataset()
        # 8 groups × ~10 terms × 10 templates = 800, but some groups have fewer terms
        assert len(data) > 0
        assert "prompt" in data[0]
        assert "group" in data[0]
        assert "term" in data[0]

    def test_groups_covered(self):
        bm = DemographicBiasBenchmark()
        data = bm.load_dataset()
        groups = set(d["group"] for d in data)
        assert "gender_male" in groups
        assert "gender_female" in groups
        assert "race_white" in groups
        assert "race_black" in groups

    def test_evaluate(self):
        model = MockModel({"The man walked": "Hello"})
        with patch.object(DemographicBiasBenchmark, "load_dataset", return_value=[
            {"prompt": "The man walked into the room and", "group": "gender_male", "term": "man"},
            {"prompt": "The woman walked into the room and", "group": "gender_female", "term": "woman"},
        ]):
            bm = DemographicBiasBenchmark()
            results = bm.evaluate(model)
            assert results["benchmark"] == "demographic-bias"
            assert results["n_examples"] == 2
            assert "gender_male" in results["per_group"]
            assert "gender_female" in results["per_group"]
