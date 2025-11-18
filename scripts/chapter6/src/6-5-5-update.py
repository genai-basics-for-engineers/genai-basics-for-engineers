from dotenv import load_dotenv; load_dotenv()
from langfuse import Langfuse
import os

NAME = "product-helpful-answer"
TYPE = "text"
MODEL = "gpt-5-nano"

NEW_TEXT = (
    "You are a Japanese assistant.\n"
    "Answer each question using only the exact expected term if it is obvious.\n"
    "Do not add explanations or extra context.\n"
    "Question: {{query}}"
)

lf = Langfuse(host=os.getenv("LANGFUSE_BASE_URL"))

prompt = lf.create_prompt(
    name=NAME,
    type=TYPE,
    prompt=NEW_TEXT,
    config={"model": MODEL},
    labels=["production"]
)

print("✅ updated:", NAME, "to version", prompt.version, "with label production")

