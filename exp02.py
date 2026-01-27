import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
assert api_key is not None, "GROQ_API_KEY is not set"
print("API key loaded successfully")

# Initialize model ONCE
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key
)

def call_chatgpt(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = llm.invoke(messages)
    return response.content

def solve_problem(question):
    prompt = f"""
Solve the following problem and provide a clear explanation.

Question:
{question}
"""
    return call_chatgpt(prompt)

print(
    solve_problem(
        "A cyclist rides 90 km in 3 hours. What is the average speed in km/h?"
    )
)

def few_shot_sentimen(text):
  prompt=f"""
  classify the sentiment as positive,Negative,or Neutral.

  Example1:
  Text: I love this product!.
  Sentiment: Positive

  Example2:
  Text: This is the worst product I have ever bought.
  Sentiment: Negative

  Example3:
  Text: This product is okay.
  Sentiment: Neutral

  Example4:
  Text: {text}
  Sentiment:
  """
  return call_chatgpt(prompt)
print(few_shot_sentimen("The movie was fantastic! I really enjoyed it."))
print(few_shot_sentimen("The food was terrible and the service was slow."))