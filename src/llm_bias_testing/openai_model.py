"""OpenAI model interface."""
from openai import OpenAI


class Model:
    """OpenAI model wrapper."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        self.model_name = model_name
        
        if api_key is None:
            try:
                from keys import OPENAI_API_KEY
                api_key = OPENAI_API_KEY
            except ImportError as e:
                raise ImportError(
                    "keys.py not found. Copy keys.py.example to keys.py and add your API keys."
                ) from e
        
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
