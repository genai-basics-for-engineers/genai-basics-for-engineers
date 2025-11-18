from dotenv import load_dotenv; load_dotenv()
from langfuse import Langfuse
from langfuse.openai import openai
import os

lf = Langfuse()
client = openai.OpenAI()

PROMPT_NAME = "product-helpful-answer"
PROMPT_TYPE = "text"

prompt = lf.get_prompt(PROMPT_NAME, type=PROMPT_TYPE)

compiled = prompt.compile(query="Langfuseは何ができますか？")

messages = compiled if isinstance(compiled, list) else [{"role": "user", "content": compiled}]

resp = client.chat.completions.create(
    model="gpt-5-nano",
    messages=messages,
)

print(resp.choices[0].message.content)

