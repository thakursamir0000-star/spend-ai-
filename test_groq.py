import os, sys, tomllib
from groq import Groq

try:
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        key = secrets.get("GROQ_API_KEY", "")
    else:
        key = os.environ.get("GROQ_API_KEY", "")

    if not key:
        print("No GROQ_API_KEY found")
        sys.exit(1)

    client = Groq(api_key=key)
    models = client.models.list()
    print(f"GROQ API connected successfully. Available models: {len(models.data)}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
