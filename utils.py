import ollama


class OllamaClient:
    def __init__(self, host, model):
        self.ollama_client = ollama.Client(host)
        self.model = model

    def generate(self, text, context):
        return self.ollama_client.generate(model=self.model, prompt=text, context=context)
        