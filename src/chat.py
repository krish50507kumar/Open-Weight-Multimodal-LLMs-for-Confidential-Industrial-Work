import ollama
class LLM:
    def __init__(self,model):
        self.model = model
    def chat(self,prompt):

        response = ollama.chat(model=self.model, messages=messages)
        reply = response['message']['content']
        return reply