from dotenv import load_dotenv; load_dotenv()
from langfuse import Langfuse
from langfuse.openai import openai

PROMPT_NAME = "product-helpful-answer"
PROMPT_TYPE = "text"
MODEL = "gpt-5-nano"
DATASET = "faq-ja-small"
RUN_NAME = "canary-eval"

lf = Langfuse()
client = openai.OpenAI()

try:
    lf.create_dataset(name=DATASET)
    lf.create_dataset_item(dataset_name=DATASET, input={"query": "返品は可能ですか？"}, expected_output="返品ポリシー")
    lf.create_dataset_item(dataset_name=DATASET, input={"query": "配送に何日かかりますか？"}, expected_output="通常は")
    lf.create_dataset_item(dataset_name=DATASET, input={"query": "サポートの連絡先は？"}, expected_output="サポート")
except Exception:
    pass

dataset = lf.get_dataset(name=DATASET)

for item in dataset.items:
    with item.run(run_name=RUN_NAME) as _root:
        prompt = lf.get_prompt(PROMPT_NAME, type=PROMPT_TYPE)
        compiled = prompt.compile(**item.input)
        messages = compiled if isinstance(compiled, list) else [{"role": "user", "content": compiled}]
        r = client.chat.completions.create(model=MODEL, messages=messages)
        out = r.choices[0].message.content or ""
        exp = item.expected_output if isinstance(item.expected_output, str) else str(item.expected_output)
        ok = bool(exp and (exp in out))
        _root.score_trace(name="dataset_accuracy", value=1.0 if ok else 0.0)
print("✅ dataset run:", RUN_NAME)

