from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

response = model.invoke(
    "SEC EDGAR nedir? Türkçe ve tek cümleyle açıkla."
)

print(response.content)