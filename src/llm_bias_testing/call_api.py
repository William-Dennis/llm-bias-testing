import ollama

from llm_bias_testing.ollama_setup import OllamaServer
from llm_bias_testing.transformers import Model as TransformerModel

LLM_MODEL = "gemma3:1b-it-qat"
PROVIDER = "ollama"


class Model:
    def __init__(self, model_name: str = LLM_MODEL, provider: str = PROVIDER):
        self.model_name = model_name
        self.provider = provider

        if provider == "ollama":
            self.setup_ollama()

        if provider == "transformers":
            self.setup_transformers()

    def predict(self, input_text: str, temperature: float = 0.0) -> str:
        if self.provider == "ollama":
            return self.predict_ollama(input_text, temperature)
        elif self.provider == "transformers":
            return self.predict_transformers(input_text, temperature)
        else:
            raise ValueError(
                f"Unknown predict setup, model: {self.model_name}, provider: {self.provider}"
            )

    def setup_ollama(self):
        print("Starting Ollama Server...")
        self.server = OllamaServer()
        self.server.start()

    def setup_transformers(self):
        model = TransformerModel()
        self.model = model

    def predict_transformers(self, input_text: str, temperature: float = 0.0):
        return self.model.predict(input_text, temperature)

    def predict_ollama(self, input_text: str, temperature: float = 0.0) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            options={
                "temperature": temperature,
            },
        )
        return response["message"]["content"]
