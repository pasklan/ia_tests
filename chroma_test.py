import os
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# Inicializa o modelo Google Generative AI com a versão especificada
model = GoogleGenerativeAI(model='gemini-2.0-flash-001')

# Caminho do arquivo PDF que será processado
pdf_path = 'laptop_manual.pdf'

# Carrega o arquivo PDF usando o PyPDFLoader
loader = PyPDFLoader(pdf_path)

# Carrega os documentos do PDF
docs = loader.load()

# Configura o divisor de texto para dividir os documentos em partes menores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Tamanho máximo de cada parte
    chunk_overlap=200,  # Sobreposição entre as partes
)

# Divide os documentos em partes menores (chunks)
chunks = text_splitter.split_documents(documents=docs)

# Imprime o número total de partes geradas
# print(chunks)

# Inicializa o modelo de embeddings do Google Generative AI para gerar vetores de texto
embedding = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',  # Especifica o modelo de embeddings a ser usado
)

# Cria um armazenamento vetorial (Vector Store) usando os documentos divididos e os embeddings
vector_store = Chroma.from_documents(
    documents=chunks,  # Documentos divididos em partes menores
    embedding=embedding,  # Modelo de embeddings para gerar vetores
    collection_name="laptop_manual",  # Nome da coleção no armazenamento vetorial
)

# Configura o armazenamento vetorial como um mecanismo de recuperação (retriever)
retriever = vector_store.as_retriever()

# Faz uma consulta ao mecanismo de recuperação com uma pergunta específica
prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {
        'context': retriever,
        'question': RunnablePassthrough(),
    }
    | prompt
    | model
    |  StrOutputParser()
)

try:
    while True:
        question = input("Pergunte: ")
        response = rag_chain.invoke(question)
        print(response)
except KeyboardInterrupt:
    exit()