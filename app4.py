import os
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("API_KEY")  # Obter a chave de API do arquivo .env
print(api_key)


model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-001')

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang='pt'
    )
)

agent_executor = create_python_agent(
    llm=model,
    tool=wikipedia_tool,
    verbose=True,
)

# Definir o prompt
prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Pesquise na web sobre "{query}" e resuma o conteúdo.'''
)

query = "Alan Turing"
prompt = prompt_template.format(query=query)

response = agent_executor.invoke(prompt)

print(response.get('output'))