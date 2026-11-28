"""
========================================================================================
Groq Active Models Inspector
========================================================================================
Lists all active models currently supported on your Groq API account.
========================================================================================
"""

import os
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY is missing from .env")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

try:
    models = client.models.list()
    print("=" * 60)
    print(" 🚀 ACTIVE GROQ MODELS AVAILABLE ON YOUR ACCOUNT:")
    print("=" * 60)
    chat_models = []
    other_models = []
    
    for m in models.data:
        mid = m.id
        if any(skip in mid for skip in ["whisper", "guard"]):
            other_models.append(mid)
        else:
            chat_models.append(mid)

    print("\n💬 Chat & Text Models:")
    for m in chat_models:
        print(f"   • {m}")

    print("\n🎧 Audio & Utility Models:")
    for m in other_models:
        print(f"   • {m}")
    print("=" * 60)

except Exception as e:
    print(f"❌ Error fetching models from Groq: {e}")
