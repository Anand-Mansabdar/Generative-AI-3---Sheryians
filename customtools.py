from langchain.tools import tool

@tool # Decorator to create a custom tool
def greeting(name: str, age: int) -> str:
  """Generate a greeting message for a user"""
  
  return f"Good morning!! My name is {name}. I am {age} years old..."

result = greeting.invoke({"name": "Anand", "age": 21})

print(result)
print(greeting.name)
print(greeting.description) # Prints the content within the doc string
print(greeting.args)