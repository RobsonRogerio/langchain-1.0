from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import os

load_dotenv()

# model = init_chat_model('groq:openai/gpt-oss-20b')
model = init_chat_model('groq:openai/gpt-oss-120b', temperature=0)
# model = init_chat_model(model='gemini-flash-latest', model_provider='google_genai')

DB_PATH = os.path.abspath("loja.sqlite")
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

system_prompt = """Você é um assistente especialista em consultas SQL.

Regras obrigatórias:
- Sempre inspecione as tabelas disponíveis e o esquema ANTES de gerar qualquer query.
- Limite resultados a no máximo 10 registros (use LIMIT 10).
- NUNCA execute comandos DML (INSERT, UPDATE, DELETE, DROP). Somente SELECT.
- Se não encontrar a informação, diga que os dados não foram encontrados.
- Responda sempre em português.
"""

agente_banco = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=tools
)