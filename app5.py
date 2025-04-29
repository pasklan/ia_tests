import os
from langchain import hub
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # Obter a chave de API do arquivo .env
if not api_key:
    raise ValueError("API_KEY não encontrada no arquivo .env")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
print(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

os.environ['GEMINI_API_KEY'] = api_key

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-001')

db = SQLDatabase.from_uri('sqlite:///ipca.db')

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
)

system_message = hub.pull('hwchase17/react')
agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

# Definir o prompt
prompt ='''
    Use as ferramentas necessárias para responder perguntas relacionadas ao histórico de IPCA ao longo do tempo.
    Responda tudo em português brasileiro.
    Perguntas: {q}
'''

prompt_template = PromptTemplate.from_template(prompt)
question = '''
Baseado nos dados históricos de IPCA desde 2004,
faça uma previsão dos valores de IPCA de cada mês futuro até o Fevereiro de 2025.
'''

output = agent_executor.invoke({
    'input': prompt_template.format(q=question),
})

print(output.get('output'))
