import os

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler


def get_langfuse_handler():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("⚠️ Langfuse no configurado")
        return None

    Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )

    langfuse = get_client()

    if not langfuse.auth_check():
        print("⚠️ Langfuse auth_check falló")
        return None

    print("✅ Langfuse conectado")

    return CallbackHandler()