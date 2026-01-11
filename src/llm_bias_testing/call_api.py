import ollama
from openai import OpenAI

from llm_bias_testing.ollama_setup import OllamaServer
from llm_bias_testing.transformers import Model as TransformerModel
from keys import OPENAI_API_KEY

LLM_MODEL = "gemma3:1b-it-qat"
PROVIDER = "ollama"


class Model:
    """Unified model interface supporting multiple LLM providers."""

    def __init__(self, model_name: str = LLM_MODEL, provider: str = PROVIDER):
        self.model_name = model_name
        self.provider = provider

        if provider == "ollama":
            self.setup_ollama()
        elif provider == "transformers":
            self.setup_transformers()
        elif provider == "openai":
            self.setup_openai()

    def predict(self, input_text: str, temperature: float = 0.0) -> str:
        """Generate prediction using configured provider."""
        if self.provider == "ollama":
            return self.predict_ollama(input_text, temperature)
        elif self.provider == "transformers":
            return self.predict_transformers(input_text, temperature)
        elif self.provider == "openai":
            return self.predict_openai(input_text, temperature)
        else:
            raise ValueError(
                f"Unknown predict setup, model: {self.model_name}, provider: {self.provider}"
            )

    def setup_ollama(self):
        """Initialize Ollama server."""
        print("Starting Ollama Server...")
        self.server = OllamaServer()
        self.server.start()

    def setup_transformers(self):
        """Initialize Transformers model."""
        model = TransformerModel()
        self.model = model

    def setup_openai(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def predict_transformers(self, input_text: str, temperature: float = 0.0):
        """Generate prediction using Transformers."""
        return self.model.predict(input_text, temperature)

    def predict_ollama(self, input_text: str, temperature: float = 0.0) -> str:
        """Generate prediction using Ollama."""
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            options={
                "temperature": temperature,
            },
        )
        return response["message"]["content"]

    def predict_openai(self, input_text: str, temperature: float = 0.0) -> str:
        """Generate prediction using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
        )
        return response.choices[0].message.content
