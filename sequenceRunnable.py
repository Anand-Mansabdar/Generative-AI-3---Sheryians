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

# Formatting the prompt
formatted_prompt = prompt.format_messages(topic="Machine Learning")

# Calling the model manually
response = model.invoke(formatted_prompt)

final_output = parser.parse(response.content)

print(final_output)