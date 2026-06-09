from abc import ABC, abstractmethod
import json
import os


class BaseBenchmark(ABC):
    name: str = "base"

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def evaluate(self, model) -> dict:
        pass

    def save_results(self, results: dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"{self.name}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    def load_results(self, output_dir: str) -> dict | None:
        results_file = os.path.join(output_dir, f"{self.name}.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                return json.load(f)
        return None
