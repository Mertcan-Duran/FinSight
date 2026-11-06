from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print("API key bulundu.")
else:
    print("API key bulunamadı.")