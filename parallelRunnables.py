from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# 1. Model
model = ChatMistralAI(model="mistral-small-2506")

# 2. Output Parser
parser = StrOutputParser()

# 3. Creating multiple prompts
short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 1-2 lines")

detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")

# 4. Input
topic = "How to post high engaging linkedin drafts for more engagement"

# 5. Creating Parallel Runnables
chain = RunnableParallel({
  "short": short_prompt | model | parser,
  "detailed": detailed_prompt | model | parser
}) # A dictionary must be converted to a ParallelRunnable using RunnableParallel

response = chain.invoke({"topic": topic})
# print(response)
print(response['short'])
print(response['detailed'])