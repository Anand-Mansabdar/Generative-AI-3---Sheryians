from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a code generator"),
  ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a helpful assistant who explains code in simple terms"),
  ("human", "Explain the following code in simple words:\n {code} ")
])

sequence = code_prompt | model | parser

sequence2 = RunnableParallel({
  "code": RunnablePassthrough(),
  "explaination": explain_prompt | model | parser
})

chain = sequence | sequence2

result = chain.invoke({"topic" : "Write a code for prime numebers in python"})

print(result["code"])
print(result["explaination"])

# The result gives us the explaination but we are never able to see what code it has generated after we pass the sequence from code_prompt -> model -> parser...

# The output of the above runnable gets lost and the output from the second part(explain_prompt) will give us the final output.. Hence to see the output of the 1st half we need RunnablePassthrough