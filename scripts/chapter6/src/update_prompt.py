from dotenv import load_dotenv; load_dotenv()
from langfuse import Langfuse

NAME = "product-helpful-answer"
TYPE = "text"
MODEL = "gpt-5-nano"

NEW_TEXT = (
    "You are a Japanese assistant.\n"
    "Answer each question using only the exact expected term if it is obvious.\n"
    "Do not add explanations or extra context.\n"
    "Question: {{query}}"
)

lf = Langfuse()

prompt = lf.create_prompt(
    name=NAME,
    type=TYPE,
    prompt=NEW_TEXT,
    config={"model": MODEL},
    labels=["stable"]  # 一発で stable ラベルを付与
)

print("✅ updated:", NAME, "to version", prompt.version, "with label stable")

