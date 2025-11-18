from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def render(self, **kwargs):
        return self.template.format(**kwargs)

# 文書要約用テンプレート
SUMMARY_TEMPLATE = PromptTemplate("""
あなたは文書要約の専門家です。

以下の文書を{length}で要約してください：

文書:
{document}

要約にあたって以下の点に注意してください：
- 重要な数値や固有名詞は必ず含める
- {audience}向けの表現を使用する
- {format}形式で出力する

要約:
""")

# サンプル文書を定義
document_text = """
OpenAIは2023年にChatGPTを公開し、世界中で数億人のユーザーに利用されるようになりました。
特に教育、ビジネス、ソフトウェア開発の分野で大きな影響を与えています。
同社はまた、GPT-4やGPT-5といった先進的なモデルを次々に発表しています。
"""

# テンプレートに値を埋め込む
prompt = SUMMARY_TEMPLATE.render(
    length="200文字程度",
    document=document_text,
    audience="経営陣",
    format="箇条書き"
)

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": prompt}],
)

print("=== プロンプト ===")
print(prompt)

print("=== 要約結果 ===")
print(response.choices[0].message.content.strip())
