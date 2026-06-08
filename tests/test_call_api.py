from unittest.mock import patch

from llm_bias_testing.call_api import Model, LLM_MODEL, PROVIDER


class TestModelInit:
    def test_model_init_default(self):
        with patch.object(Model, "setup_ollama"):
            model = Model()
        assert model.model_name == LLM_MODEL
        assert model.provider == PROVIDER

    def test_model_init_custom(self):
        model = Model(model_name="custom-model", provider="custom")
        assert model.model_name == "custom-model"
        assert model.provider == "custom"
