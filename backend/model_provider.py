"""Request-scoped model selection; credentials stay on the backend."""
import json
import os
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

DEFAULT_MODEL = "llama3.1:8b"
GPT_MODEL = "gpt-4.1-mini"
ModelId = Literal["llama3.1:8b", "gpt-4.1-mini"]


def validate_model(model: str):
    if model not in (DEFAULT_MODEL, GPT_MODEL):
        raise ValueError("Unsupported model. Choose llama3.1:8b or gpt-4.1-mini.")
    if model == GPT_MODEL and not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("GPT requires OPENAI_API_KEY in the project's .env file. Restart the backend after setting it.")


def available_models():
    return [
        {"id": DEFAULT_MODEL, "label": "Llama 3.1 8B (Ollama)", "configured": True},
        {"id": GPT_MODEL, "label": "GPT-4.1 mini (OpenAI)",
         "configured": bool(os.getenv("OPENAI_API_KEY", "").strip())},
    ]


class GPTClient:
    """Expose the same string invoke/stream interface as OllamaLLM."""
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self):
        validate_model(GPT_MODEL)

    def _client(self):
        return httpx.Client(
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
            timeout=httpx.Timeout(120, connect=15),
        )

    def _payload(self, prompt, stream=False):
        return {"model": GPT_MODEL, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.65, "max_completion_tokens": 512, "stream": stream}

    @staticmethod
    def _check(response):
        # Never expose a provider response body: auth errors can echo credentials.
        if response.is_success:
            return
        messages = {
            401: "OpenAI rejected the API key. Update OPENAI_API_KEY and restart the backend.",
            403: "OpenAI access denied. Check account and model access.",
            404: "GPT model unavailable for this OpenAI project.",
            429: "OpenAI quota or rate limit reached. Check billing or try again later.",
        }
        raise RuntimeError(messages.get(response.status_code, "OpenAI request failed. Try again later."))

    def invoke(self, prompt):
        try:
            with self._client() as client:
                response = client.post(self.url, json=self._payload(prompt))
                self._check(response)
                return response.json()["choices"][0]["message"]["content"] or ""
        except httpx.RequestError:
            raise RuntimeError("Cannot reach OpenAI. Check your internet connection and try again.") from None

    def stream(self, prompt):
        try:
            with self._client() as client:
                with client.stream("POST", self.url, json=self._payload(prompt, True)) as response:
                    self._check(response)
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            return
                        event = json.loads(data)
                        if "error" in event:
                            raise RuntimeError("OpenAI generation failed. Please try again.")
                        choices = event.get("choices", [])
                        if choices:
                            token = choices[0].get("delta", {}).get("content")
                            if token:
                                yield token
                    raise RuntimeError("OpenAI stream ended early. Please try again.")
        except httpx.RequestError:
            raise RuntimeError("Cannot reach OpenAI. Check your internet connection and try again.") from None
