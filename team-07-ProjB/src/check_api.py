from dotenv import load_dotenv
import os

load_dotenv()  # reads .env from current working directory
print("OPENAI_API_KEY set?", bool(os.getenv("OPENAI_API_KEY")))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))