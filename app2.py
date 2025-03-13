import os
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_google_genai import GoogleGenerativeAI

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave de API do arquivo .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API_KEY não encontrada no arquivo .env")
os.environ['GEMINI_API_KEY'] = api_key

model = GoogleGenerativeAI(model='gemini-1.5-pro-002')

python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Shell Python, use para executar código Python. Execute apenas códigos Python válidos. Se precisar obter o retorno do código, use a função "print(...)"',
    func=python_repl.run,
)

agent_executor = create_python_agent(
    llm=model,
    tool=python_repl_tool,
    verbose=True,
    handle_parsing_errors=True,
)

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Resolva o problema: "{query}".
    '''
)

query = r"quanto é 20% de 3000?"
prompt = prompt_template.format(query=query)

try:
    response = agent_executor.invoke(prompt)
    print(response.get('output'))
except Exception as e:
    print(f"Erro ao analisar a saída do modelo: {e}")
