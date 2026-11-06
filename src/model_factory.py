import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


load_dotenv()


def get_chat_model():
    provider = os.getenv(
        "FINSIGHT_MODEL_PROVIDER",
        "ollama",
    ).strip().lower()

    if provider == "ollama":
        return ChatOllama(
            model="qwen3:4b",
            temperature=0,
        )

    if provider == "groq":
        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is required "
                "when FINSIGHT_MODEL_PROVIDER=groq"
            )

        return ChatOpenAI(
            model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            ),
            base_url=(
                "https://api.groq.com/openai/v1"
            ),
            api_key=api_key,
            temperature=0,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0,
        )

    raise ValueError(
        f"Unsupported FINSIGHT_MODEL_PROVIDER: "
        f"{provider}"
    )