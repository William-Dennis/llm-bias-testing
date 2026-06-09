MODELS = {
    "smollm-135m": {
        "ollama_tag": "smollm:135m",
        "params": 135_000_000,
        "release_date": "2024-07",
        "family": "huggingface",
        "architecture": "decoder-only",
    },
    "smollm-360m": {
        "ollama_tag": "smollm:360m",
        "params": 360_000_000,
        "release_date": "2024-07",
        "family": "huggingface",
        "architecture": "decoder-only",
    },
    "smollm2-135m": {
        "ollama_tag": "smollm2:135m",
        "params": 135_000_000,
        "release_date": "2024-11",
        "family": "huggingface",
        "architecture": "decoder-only",
    },
    "smollm2-360m": {
        "ollama_tag": "smollm2:360m",
        "params": 360_000_000,
        "release_date": "2024-11",
        "family": "huggingface",
        "architecture": "decoder-only",
    },
    "qwen25-05b": {
        "ollama_tag": "qwen2.5:0.5b",
        "params": 500_000_000,
        "release_date": "2024-09",
        "family": "alibaba",
        "architecture": "decoder-only",
    },
    "qwen25-15b": {
        "ollama_tag": "qwen2.5:1.5b",
        "params": 1_500_000_000,
        "release_date": "2024-09",
        "family": "alibaba",
        "architecture": "decoder-only",
    },
    "qwen35-08b": {
        "ollama_tag": "qwen3.5:0.8b",
        "params": 800_000_000,
        "release_date": "2025-05",
        "family": "alibaba",
        "architecture": "decoder-only",
    },
    "llama32-1b": {
        "ollama_tag": "llama3.2:1b",
        "params": 1_000_000_000,
        "release_date": "2024-09",
        "family": "meta",
        "architecture": "decoder-only",
    },
    "tinyllama": {
        "ollama_tag": "tinyllama",
        "params": 1_100_000_000,
        "release_date": "2023-12",
        "family": "meta",
        "architecture": "decoder-only",
    },
    "stablelm2-16b": {
        "ollama_tag": "stablelm2:1.6b",
        "params": 1_600_000_000,
        "release_date": "2024-01",
        "family": "stability",
        "architecture": "decoder-only",
    },
    "gemma3-1b": {
        "ollama_tag": "gemma3:1b-it-qat",
        "params": 1_000_000_000,
        "release_date": "2025-03",
        "family": "google",
        "architecture": "decoder-only",
    },
}


def get_model(name: str) -> dict:
    """Get model config by name. Raises KeyError if not found."""
    return MODELS[name]


def list_models() -> list[str]:
    """List all registered model names."""
    return list(MODELS.keys())


def models_by_family(family: str) -> dict:
    """Filter models by family."""
    return {k: v for k, v in MODELS.items() if v["family"] == family}