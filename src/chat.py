import ollama


class LLM:
    def __init__(self, model: str):
        self.model = model

    def chat(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        # Handle both dict and object-style responses across Ollama SDK versions
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        # newer SDK: ChatResponse object
        msg = getattr(response, "message", None)
        if msg is not None:
            return getattr(msg, "content", "") or str(response)
        return str(response)