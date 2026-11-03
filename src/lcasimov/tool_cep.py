from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

import requests

load_dotenv()

@wrap_tool_call
async def tratar_erros(request, handler):
    try:
        return await handler(request)
    except Exception as e:
        tool_call_id = request.tool_call["id"]
        return ToolMessage(
            content=f"Erro ao executar {request.tool_call['name']}: {str(e)}. Verifique os parametros enviados",
            tool_call_id=tool_call_id
    )

class CepInput(BaseModel):
    cep: str = Field(..., description="O cep a ser buscado. 8 digítos no padrão brasileiro")
    @field_validator("cep")
    @classmethod
    def validar_cep(cls, v: str) -> str:
        cep_limpo = v.replace("-", "").strip()
        if len(cep_limpo) != 8 or not cep_limpo.isdigit():
            raise ValueError("O cep deve ter 8 dígitos numéricos.")
        return cep_limpo

# model = init_chat_model('groq:openai/gpt-oss-20b')
model = init_chat_model('groq:openai/gpt-oss-120b')
# model = init_chat_model(model='gemini-flash-latest', model_provider='google_genai')

@tool(args_schema=CepInput)
def buscar_cep(cep:str) -> str:
    """
    Busca ceps no Brasil
    Args:
        cep: O cep a ser buscado
    """
    url = f"https://viacep.com.br/ws/{cep}/json/"
    response = requests.get(url)
    data = response.json()
    return data

agent_cep = create_agent(
    model=model,
    system_prompt='Você é um assistente que responde perguntas sobre o que você sabe. Use a ferramenta de busca quando precisar de informação atual. Se não souber não invente respostas',
    tools=[buscar_cep],
    middleware=[tratar_erros]
)
