import pytest

from llm_bias_testing.registry import MODELS, get_model, list_models, models_by_family


class TestGetModel:
    def test_get_model_valid(self):
        model = get_model("smollm-135m")
        assert model["ollama_tag"] == "smollm:135m"
        assert model["params"] == 135_000_000
        assert model["family"] == "huggingface"

    def test_get_model_invalid(self):
        with pytest.raises(KeyError):
            get_model("nonexistent-model")


class TestListModels:
    def test_list_models_returns_all(self):
        models = list_models()
        assert len(models) == len(MODELS)
        assert "smollm-135m" in models
        assert "gemma3-1b" in models

    def test_list_models_order(self):
        models = list_models()
        assert models == list(MODELS.keys())


class TestModelsByFamily:
    def test_filter_by_family(self):
        hf = models_by_family("huggingface")
        assert all(v["family"] == "huggingface" for v in hf.values())
        assert "smollm-135m" in hf
        assert "smollm2-360m" in hf

    def test_filter_nonexistent_family(self):
        result = models_by_family("nonexistent")
        assert result == {}


VALID_ARCHS = {"decoder-only", "hybrid-conv-attn"}


class TestRegistryCompleteness:
    def test_all_models_have_required_fields(self):
        required = {"ollama_tag", "params", "release_date", "family", "architecture"}
        for name, config in MODELS.items():
            assert required.issubset(config.keys()), f"Model {name} missing fields: {required - set(config.keys())}"

    def test_all_params_are_positive(self):
        for name, config in MODELS.items():
            assert config["params"] > 0, f"Model {name} has non-positive params"

    def test_all_models_decoder_only(self):
        for name, config in MODELS.items():
            assert config["architecture"] in VALID_ARCHS, f"Model {name} has unexpected architecture: {config['architecture']}"
