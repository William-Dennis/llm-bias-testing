"""OpenAI model interface."""
from openai import OpenAI
from keys import OPENAI_API_KEY


class Model:
    """OpenAI model wrapper."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = OPENAI_API_KEY):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def predict(self, input_text: str, temperature: float = 0.0) -> str:
        """Generate prediction using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
        )
        if not response.choices:
            raise ValueError("OpenAI returned no response choices")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned None for message content")
        return content
