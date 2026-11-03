from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

from dotenv import load_dotenv

load_dotenv()

# model = init_chat_model('groq:openai/gpt-oss-20b')
model = init_chat_model('groq:openai/gpt-oss-120b')
# model = init_chat_model(model='gemini-flash-latest', model_provider='google_genai')

@tool
def converter_temperatura(celsius: float) -> str:
    """
    Converte uma temperatura de Celsius para Fahrenheit
    Use quando o usuario perguntar sobre conversão de temperatura
    Args:
        celsius: Temperatura em graus Celsius
    """
    fahrenheit = (celsius * 9/5 + 32)
    return f'{celsius}°C é igual a {fahrenheit}°F'

agent_temperature = create_agent(
    model=model,
    system_prompt='Você é um assistente que responde perguntas sobre o que você sabe. Use a ferramenta de busca quando precisar de informação atual. Se não souber não invente respostas',
    tools=[converter_temperatura]
)

