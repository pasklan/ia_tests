import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
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

prompt = '''
Como assistente financeiro pessoal, que responderá as perguntas dando dicas financeiras, e de investimento.
Responda tudo em português brasileiro.
Pergutas: {q}
'''

prompt_template = PromptTemplate.from_template(prompt)

# Tool 1 - Python REPL
python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Um shell Python. Use isso para executar código Python. Execute apenas códigos Python válidos. Se precisar obter o retorno do código, use a função "print(...)"'
    'Se você precisar obter o retorno do código, use a função "print(...)"'
    'Use para realizar cálculos financeiros necessários para responder as perguntas financeiras.',
    func=python_repl.run,
)

# Tool 2 - DuckDuckGo Search
search = DuckDuckGoSearchRun()
duckduck_go_tool = Tool(
    name='Buscar DuckDuckGo',
    description='Use isso para pesquisar na web sobre informações e dicas de economia e opções de investimento.'
                'Sempre deve pesquisar na internet as melhores dicas usando esta ferramenta, não'
                'responda diretamente. Sua resposta deve informar que há elementos pesquisados na internet.',
    func=search.run,
)

react_instructions = hub.pull('hwchase17/react')

tools = [python_repl_tool, duckduck_go_tool]

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt = react_instructions,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

question = '''
Minha renda é de R$10000 por m6es, o total de minhas despesas é de R$8500 mais 1000 de aluguel.
Quais dicas de investimento você me dá?
'''

output = agent_executor.invoke(
    {'input': prompt_template.format(q=question)}
)

print(output.get('output'))

