from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def contextual_translate(text, context=""):
    """文脈を考慮した翻訳関数"""
    prompt = f"""
以下の文章を日本語へ翻訳してください。

文脈情報: {context}
翻訳対象: {text}

翻訳時の注意点:
- 原文の意図とニュアンスを保持する
- 専門用語は適切に扱う
"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# 使用例
text = "Before starting the maintenance work, please disconnect the power supply, check that all moving parts have come to a complete stop, and make sure the surrounding area is free from obstacles. Always wear protective gloves and safety goggles, as unexpected sparks or sharp edges may cause injury."

# ビジネス風に翻訳
context_novel = "ビジネス文書風で翻訳してください。"
result_novel = contextual_translate(text, context_novel)
print("=== ビジネス文書風翻訳 ===")
print(result_novel)

# カジュアル風に翻訳
context_tech = "カジュアル風で翻訳してください。"
result_tech = contextual_translate(text, context_tech)
print("\n=== カジュアル風翻訳 ===")
print(result_tech)
