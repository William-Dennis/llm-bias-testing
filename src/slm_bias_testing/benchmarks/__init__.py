from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any


class BaseBenchmark(ABC):
    name: str = "base"

    @abstractmethod
    def load_dataset(self) -> list[Any]: ...

    @abstractmethod
    def evaluate(self, model: Any, max_samples: int | None = None) -> dict[str, Any]: ...

    def save_results(self, results: dict[str, Any], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"{self.name}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    def load_results(self, output_dir: str) -> dict[str, Any] | None:
        results_file = os.path.join(output_dir, f"{self.name}.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return None
