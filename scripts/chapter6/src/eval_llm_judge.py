from dotenv import load_dotenv; load_dotenv()
from langfuse import Langfuse
from langfuse.openai import openai
import json, re

PROMPT_NAME = "product-helpful-answer"
PROMPT_TYPE = "text"
MODEL = "gpt-5-nano"
JUDGE_MODEL = "gpt-5-nano"
PASS_THRESHOLD = 0.7
DATASET = "faq-ja-small"
RUN_NAME = "llm-judge-v1"

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
    with item.run(run_name=RUN_NAME) as root:
        p = lf.get_prompt(PROMPT_NAME, type=PROMPT_TYPE)
        compiled = p.compile(**item.input)
        messages = compiled if isinstance(compiled, list) else [{"role":"user","content":compiled}]
        r = client.chat.completions.create(model=MODEL, messages=messages)
        pred = r.choices[0].message.content or ""
        expected = item.expected_output if isinstance(item.expected_output,str) else str(item.expected_output)
        rubric = "You are a strict Japanese grader. Score how well the candidate answer satisfies the expected intent. Return JSON with keys: score (0.0-1.0), reasoning."
        jm = [
            {"role":"system","content":rubric},
            {"role":"user","content":f"Expected:\n{expected}\n\nAnswer:\n{pred}\n\nReturn JSON only."}
        ]
        jr = client.chat.completions.create(model=JUDGE_MODEL, messages=jm)
        txt = jr.choices[0].message.content or "{}"
        m = re.search(r"\{.*\}", txt, re.S)
        js = json.loads(m.group(0)) if m else {"score":0.0}
        score = float(js.get("score",0.0))
        root.score_trace(name="judge_score", value=score)
        root.score_trace(name="judge_pass", value=1.0 if score>=PASS_THRESHOLD else 0.0)
print("done:", RUN_NAME)

