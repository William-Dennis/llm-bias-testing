import ollama

LLM_MODEL = "gemma3:latest"


class Model:
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name

    def predict(self, input_text: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            options={
                "temperature": 0.0,
            },
        )
        return response["message"]["content"]
