import sqlite3
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import SummarizationMiddleware

conn = sqlite3.connect('checkpoints.db', check_same_thread=False)

checkpoint = SqliteSaver(conn)

load_dotenv()

# model = init_chat_model('groq:openai/gpt-oss-20b')
model = init_chat_model('groq:openai/gpt-oss-120b')
# model = init_chat_model(model='gemini-flash-latest', model_provider='google_genai')

agent_asimov = create_agent(
    model=model,
    system_prompt='Você é um assistente que responde perguntas sobre o que você sabe. Use a ferramenta de busca quando precisar de informação atual. Se não souber não invente respostas',
    tools=[TavilySearch(max_results=2)],
    checkpointer = checkpoint,
    middleware=[SummarizationMiddleware(
        model=model,
        trigger=('messages', 6),
        keep=('messages', 2)
    )]
)

config = {'configurable': {'thread_id': '3_summarize'}}

print('Agente em funcionamento: ')
while True:
    pergunta = input('Faça sua pergunta (ou "sair"): ')
    if pergunta.strip().lower() in ('sair', 'exit'):
        break
    resposta = agent_asimov.invoke({'messages': [{"role": "user", "content": pergunta}]}, config)
    print(f"Agente: {resposta['messages'][-1].content}")


