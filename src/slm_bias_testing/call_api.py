import logging
import time

import ollama

from slm_bias_testing.ollama_setup import OllamaServer
from slm_bias_testing.transformers import Model as TransformerModel

logger = logging.getLogger(__name__)

LLM_MODEL = "gemma3:1b-it-qat"
PROVIDER = "ollama"

_ollama_client = ollama.Client(timeout=300)
_ollama_server = None


def _ensure_ollama():
    """Check if Ollama is responding; restart if not."""
    global _ollama_server
    try:
        _ollama_client.list()
        return
    except Exception:
        logger.warning("Ollama not responding, restarting...")
        if _ollama_server is not None:
            try:
                _ollama_server.stop()
            except Exception:
                pass
        _ollama_server = OllamaServer(kill_existing=True)
        _ollama_server.start()
        logger.info("Ollama restarted")


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
        _ensure_ollama()

    def setup_transformers(self):
        model = TransformerModel()
        self.model = model

    def predict_transformers(self, input_text: str, temperature: float = 0.0) -> str:
        return self.model.predict(input_text, temperature)

    def predict_ollama(self, input_text: str, temperature: float = 0.0) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = _ollama_client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": input_text}],
                    options={
                        "temperature": temperature,
                    },
                )
                return response["message"]["content"]
            except Exception as e:
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                if attempt < max_retries - 1:
                    _ensure_ollama()
                    time.sleep(2)
                else:
                    raise
