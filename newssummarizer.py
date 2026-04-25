from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults

load_dotenv()

search_tool = TavilySearchResults(max_result=5)

model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
  """
  You are a helpful assistant that summarizes the following news into clear bullet points
  {news}
""")

runnable = prompt | model | StrOutputParser()

news_result = search_tool.run("What the current situation of IPL 2026")

result = runnable.invoke({"news": news_result})

print(result)