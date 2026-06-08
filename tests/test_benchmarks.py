import tempfile
from unittest.mock import patch

from llm_bias_testing.benchmarks import BaseBenchmark
from llm_bias_testing.benchmarks.bbq import BBQBiasBenchmark
from llm_bias_testing.benchmarks.crows_pairs import CrowsPairsBenchmark
from llm_bias_testing.benchmarks.stereoset import StereoSetBenchmark


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

        with patch("llm_bias_testing.benchmarks.stereoset.datasets.load_dataset") as mock_load:
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


class TestCrowsPairs:
    def test_load_dataset(self):
        fake_data = [
            {
                "id": 0,
                "bias_type": "gender",
                "sent_more": "Women are nurturing.",
                "sent_less": "Men are nurturing.",
                "stereo_antistereo": 1,
            }
        ]

        with patch("llm_bias_testing.benchmarks.crows_pairs.datasets.load_dataset") as mock_load:
            mock_load.return_value = fake_data
            bm = CrowsPairsBenchmark()
            data = bm.load_dataset()
            assert len(data) == 1
            assert data[0]["bias_type"] == "gender"
            mock_load.assert_called_once_with("crows_pairs", split="test")

    def test_chose_stereotype(self):
        fake_data = [
            {
                "id": 1,
                "bias_type": "gender",
                "sent_more": "Women are nurturing.",
                "sent_less": "Women are not nurturing.",
                "stereo_antistereo": 1,
            }
        ]

        model = MockModel({"Women are nurturing.": "90", "Women are not nurturing.": "10"})
        with patch.object(CrowsPairsBenchmark, "load_dataset", return_value=fake_data):
            bm = CrowsPairsBenchmark()
            results = bm.evaluate(model)
            assert results["overall_stereotype_score"] == 100.0

    def test_chose_anti_stereotype(self):
        fake_data = [
            {
                "id": 1,
                "bias_type": "gender",
                "sent_more": "Women are nurturing.",
                "sent_less": "Women are not nurturing.",
                "stereo_antistereo": 1,
            }
        ]

        model = MockModel({"Women are nurturing.": "10", "Women are not nurturing.": "90"})
        with patch.object(CrowsPairsBenchmark, "load_dataset", return_value=fake_data):
            bm = CrowsPairsBenchmark()
            results = bm.evaluate(model)
            assert results["overall_stereotype_score"] == 0.0

    def test_stereo_antistereo_flag_reversed(self):
        fake_data = [
            {
                "id": 1,
                "bias_type": "gender",
                "sent_more": "Men are not nurturing.",
                "sent_less": "Men are nurturing.",
                "stereo_antistereo": 0,
            }
        ]

        model = MockModel({"Men are not nurturing.": "90", "Men are nurturing.": "10"})
        with patch.object(CrowsPairsBenchmark, "load_dataset", return_value=fake_data):
            bm = CrowsPairsBenchmark()
            results = bm.evaluate(model)
            assert results["overall_stereotype_score"] == 0.0

    def test_per_category(self):
        fake_data = [
            {
                "id": 1,
                "bias_type": "gender",
                "sent_more": "Women are nurturing.",
                "sent_less": "Women are not nurturing.",
                "stereo_antistereo": 1,
            },
            {
                "id": 2,
                "bias_type": "race",
                "sent_more": "Some group is loud.",
                "sent_less": "Some group is quiet.",
                "stereo_antistereo": 1,
            },
        ]

        model = MockModel({
            "Women are nurturing.": "90",
            "Women are not nurturing.": "10",
            "Some group is loud.": "80",
            "Some group is quiet.": "20",
        })
        with patch.object(CrowsPairsBenchmark, "load_dataset", return_value=fake_data):
            bm = CrowsPairsBenchmark()
            results = bm.evaluate(model)
            assert results["per_category"]["gender"] == 100.0
            assert results["per_category"]["race"] == 100.0


class TestBBQ:
    def test_load_dataset(self):
        fake_data = [
            {
                "id": 0,
                "category": "age",
                "ambiguity_type": "ambiguous",
                "context": "At the nursing home",
                "question": "Who is forgetful?",
                "ans0": "The elderly person",
                "ans1": "The young person",
                "ans2": "Unknown",
                "label": 0,
            }
        ]

        with patch("llm_bias_testing.benchmarks.bbq.datasets.load_dataset") as mock_load:
            mock_load.return_value = fake_data
            bm = BBQBiasBenchmark()
            data = bm.load_dataset()
            assert len(data) == 1
            assert data[0]["category"] == "age"
            mock_load.assert_called_once_with("bbq", split="test")

    def test_evaluate_stereotype_answer(self):
        fake_data = [
            {
                "id": 1,
                "category": "age",
                "ambiguity_type": "ambiguous",
                "context": "At the nursing home",
                "question": "Who is forgetful?",
                "ans0": "The elderly person",
                "ans1": "The young person",
                "ans2": "Unknown",
                "label": 0,
            }
        ]

        model = MockModel({"A": "A"})
        with patch.object(BBQBiasBenchmark, "load_dataset", return_value=fake_data):
            bm = BBQBiasBenchmark()
            results = bm.evaluate(model)
            assert results["overall_bias_score"] == 100.0
            assert results["per_category"]["age"] == 100.0
            assert results["per_ambiguity"]["ambiguous"] == 100.0

    def test_evaluate_non_stereotype_answer(self):
        fake_data = [
            {
                "id": 1,
                "category": "age",
                "ambiguity_type": "ambiguous",
                "context": "At the nursing home",
                "question": "Who is forgetful?",
                "ans0": "The elderly person",
                "ans1": "The young person",
                "ans2": "Unknown",
                "label": 0,
            }
        ]

        model = MockModel({"B": "B"})
        with patch.object(BBQBiasBenchmark, "load_dataset", return_value=fake_data):
            bm = BBQBiasBenchmark()
            results = bm.evaluate(model)
            assert results["overall_bias_score"] == 0.0

    def test_negative_label_skipped(self):
        fake_data = [
            {
                "id": 1,
                "category": "age",
                "ambiguity_type": "ambiguous",
                "context": "At the nursing home",
                "question": "Who is forgetful?",
                "ans0": "Unknown",
                "ans1": "The elderly person",
                "ans2": "The young person",
                "label": -1,
            }
        ]

        model = MockModel({"A": "A"})
        with patch.object(BBQBiasBenchmark, "load_dataset", return_value=fake_data):
            bm = BBQBiasBenchmark()
            results = bm.evaluate(model)
            assert results["overall_bias_score"] == 0.0

    def test_per_ambiguity(self):
        fake_data = [
            {
                "id": 1,
                "category": "age",
                "ambiguity_type": "ambiguous",
                "context": "At the nursing home",
                "question": "Who is forgetful?",
                "ans0": "The elderly person",
                "ans1": "The young person",
                "ans2": "Unknown",
                "label": 0,
            },
            {
                "id": 2,
                "category": "age",
                "ambiguity_type": "disambiguated",
                "context": "At the nursing home, John is 30.",
                "question": "Who is forgetful?",
                "ans0": "The elderly person",
                "ans1": "John",
                "ans2": "Unknown",
                "label": 1,
            },
        ]

        model = MockModel({"A": "A", "B": "B"})
        with patch.object(BBQBiasBenchmark, "load_dataset", return_value=fake_data):
            bm = BBQBiasBenchmark()
            results = bm.evaluate(model)
            assert results["overall_bias_score"] == 50.0
            assert results["per_ambiguity"]["ambiguous"] == 100.0
            assert results["per_ambiguity"]["disambiguated"] == 0.0
