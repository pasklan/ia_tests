import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API_KEY não encontrada no arquivo .env")

os.environ['GEMINI_API_KEY'] = api_key

model = GoogleGenerativeAI(model='gemini-2.0-flash-001')

embedding = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',
)
persist_directory = 'db'

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name="carros"
)

retriever = vector_store.as_retriever()

system_prompt = '''
Use o contexto para responder as perguntas.
Contexto: {context}
'''

prompt = ChatPromptTemplate.from_messages(
    {
        ('system', system_prompt),
        ('human', '{input}'),
    }
)

question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)

chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

query = 'Qual o carro mais novo?'

response = chain.invoke(
    {'input': query}
)
print(response)