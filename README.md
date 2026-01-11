# LLM Bias Testing

A simple repo for the analysis of bias on LLMs

## Setup

1. Copy the example keys file and add your API keys:
   ```bash
   cp keys.py.example keys.py
   # Edit keys.py with your actual API keys
   ```

2. Install dependencies (if using uv):
   ```bash
   uv sync
   ```

## Usage

The Model class supports multiple providers: `ollama`, `transformers`, and `openai`.

### Using OpenAI

```python
from llm_bias_testing.call_api import Model

# Initialize with OpenAI provider
model = Model(model_name="gpt-3.5-turbo", provider="openai")

# Generate predictions
response = model.predict("Your prompt here", temperature=0.7)
print(response)
```

### Using Ollama

```python
model = Model(model_name="gemma3:1b-it-qat", provider="ollama")
response = model.predict("Your prompt here")
```

### Using Transformers

```python
model = Model(provider="transformers")
response = model.predict("Your prompt here")
```

## Example: Gender in CV Screening

![Score Distribution by Name](plots/score_distribution_by_name.png)

![Score Distribution by University](plots/score_distribution_by_university.png)