from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from keys import HF_token


class Model:
    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        token: str = HF_token,
        device: str = None,
    ):
        # Use provided device string or default to cuda if available else cpu
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token = token  # save token if needed later
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(
            self.device
        )

        self.model.eval()
        if self.device.type == "cuda":
            self.model.half()  # Use FP16 only on GPU for speed
        self.model_name = model_name

    def predict(
        self, input_text: str, temperature: float = 0.0, max_new_tokens: int = 50
    ) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
