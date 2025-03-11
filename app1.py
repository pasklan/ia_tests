import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave de API do arquivo .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API_KEY não encontrada no arquivo .env")

os.environ['GEMINI_API_KEY'] = api_key

model = GoogleGenerativeAI(model='gemini-2.0-flash-001')

response = model.invoke(
    input='Quem foi Alan Turing?',
    temperature=1,
    max_tokens=100,
)

print(response)