import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

# Obtém a chave da API do arquivo .env
api_key = os.getenv("GEMINI_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    # Lança um erro se a chave da API não for encontrada
    raise ValueError("API_KEY não encontrada no arquivo .env")

# Define a chave da API como uma variável de ambiente
os.environ['GEMINI_API_KEY'] = api_key
# os.environ['OPENAI_API_KEY'] = openai_api_key

pdf_path = 'laptop_manual.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(docs)

persist_directory = 'db'

embeddings = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name='laptop_manual',
)

