from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
import os

load_dotenv()


# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

# initialize llm
llm = OpenAI(model=os.getenv("OPENAI_API_MODEL_NAME"))

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
agent.chat("What is 2123 * 215123")
