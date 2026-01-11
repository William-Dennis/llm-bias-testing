
import os
import ollama

LLM_MODEL = "gemma3:1b-it-qat"

class Model:
    def __init__(self, model_name: str = LLM_MODEL, host: str = "127.0.0.1:11434"):
        self.model_name = model_name
        self.host = host

    def predict(self, input_text: str, temperature: float = 0.0) -> str:
        # Set environment variable for the duration of this call
        old_host = os.environ.get("OLLAMA_HOST")
        os.environ["OLLAMA_HOST"] = self.host

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": input_text}],
                options={"temperature": temperature},
            )
        finally:
            # Restore old value
            if old_host is None:
                del os.environ["OLLAMA_HOST"]
            else:
                os.environ["OLLAMA_HOST"] = old_host

        return response["message"]["content"]
