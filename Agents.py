import os
import requests # Whenever we want to hit online APIs
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print

load_dotenv()

# 1. Creating a weather tool - To Fetch Weather of a City

@tool
def get_weather(city: str) -> str:
  """ Function to fetch the current weather of a city """
  API_KEY = os.getenv("OPENWEATHER_API_KEY")
  url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
  
  response = requests.get(url)
  data =response.json()
  
  print("Data:", data)
  
  if response.status_code != 200:
    return f"Error: {data.get('message', 'Could not fetch weather')}"
  
  temp = data["main"]["temp"]
  desc = data["weather"][0]["description"]
  
  return f"Weather in {city}: {desc}, {temp}°C"

# print(get_weather.invoke("Hyderabad"))



# 2. Tool using Tavily for fetching city news
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
  """Get latest news about the city"""
  response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
  results = response.get("results", [])
    
  if not results:
        return f"No news found for {city}"
    
  news_list = []
    
  for r in results:
    title = r.get("title", "No title")
    url = r.get("url", "")
    snippet = r.get("content", "")
        
    news_list.append(
      f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
    )
    
  return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

print(get_news.invoke("Hyderabad"))

# 3. Model Creation
llm = ChatMistralAI(model="mistral-small-2506")

tools = {
  "get_weather": get_weather,
  "get_news": get_news
}

llm_with_tool = llm.bind_tools([get_weather, get_news])

# Agent loop
history = []

print("City Intelligence System")
print("Type 0 to EXIT")

while True:
  user_input = input("You: ")
  if user_input == "0":
    break
  history.append(HumanMessage(content=user_input))
  
  while True:
    result = llm_with_tool.invoke(history)
    history.append(result)
    
    if result.tool_calls:
      for tool_call in result.tool_calls:
        tool_name = tool_call['name']
        
        confirm = input(f"Agent wants to call {tool_name}... Approve with Yes/No: ")
        
        if confirm.lower() == "no":
          print("Tool call permission denied. Cannot proceed further")
          break
        
        # Executing Tool
        tool_result = tools[tool_name].invoke(tool_call)
        
        history.append(ToolMessage(
          content=tool_result,
          tool_call_id = tool_call['id']
        ))
      continue
    else:
      print(f"\nFinal Answer:\n")
      print(result.content)
      print("\n"+"="*50+"\n")
      break      