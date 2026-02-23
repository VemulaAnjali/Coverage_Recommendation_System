from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv(override=True)

api_key = os.getenv("MISTRAL_API_KEY")
print("Loaded key length:", len(api_key))

client = Mistral(api_key=api_key)

resp = client.chat.complete(
    model="mistral-large-2411",
    messages=[{"role": "user", "content": "Say hello."}],
    max_tokens=10,
)

print("Model reply:", resp.choices[0].message.content)
