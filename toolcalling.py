from rich import print
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Creating a tool - Getting the length of a sentence

@tool
def get_length(statement : str) -> int:
  """The above tool returns the number of characters or length of the provided sentence or words inclusive of empty spaces and special characters"""
  
  return len(statement)

tools = {
  "get_length": get_length
}

llm = ChatMistralAI(model="mistral-small-2506")

# 2. Binding tool with ChatMistralAI
llm_binding = llm.bind_tools([get_length])

history = []

query = HumanMessage("Return the number of characters in the given sentence: 'My name is Anand Mansabdar' ")

history.append(query)
# print(history)

result = llm_binding.invoke(history)
history.append(result)

if result.tool_calls:
  print(result.tool_calls[0])
  tool_name = result.tool_calls[0]["name"]
  tool_result = tools[tool_name].invoke(result.tool_calls[0])
  history.append(tool_result)
  # print(history)
  
  
result = llm_binding.invoke(history)
print(result.content)