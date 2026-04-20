from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Creating a prompt template
prompt = ChatMessagePromptTemplate.from_template(
  "Explain {topic} in simple terms. Explain as if you were explaining it to a 5 year old"
)

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output Parser
parser = StrOutputParser()

# 4. Creating runnables and connecting runnables (using | (pipe) operator)
chain = prompt | model | parser

result = chain.invoke(topic={"Machine Learning"})

print(result)