from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

def run_baseline(user_question: str):
    response = model.invoke(user_question)
    return response.content