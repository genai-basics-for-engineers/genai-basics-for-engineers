from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_translation(original: str, translated: str) -> dict:
    """翻訳品質の自動評価"""
    evaluation_prompt = f"""
以下の翻訳を5段階（1:低品質 〜 5:高品質）で評価し、JSON形式で結果を返してください。

評価項目:
1. accuracy: 意味の正確性（1-5）
2. fluency: 表現の自然さ（1-5）
3. completeness: 情報の完全性（1-5）
4. suggestions: 改善提案（文字列）

原文: {original}
翻訳: {translated}

回答例:
{{
    "accuracy": 4,
    "fluency": 5,
    "completeness": 4,
    "suggestions": "より自然な表現にするため、〜を〜に変更することを提案します。"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0
        )

        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"評価エラー: {str(e)}"}

# 使用例
original = "The system requires immediate attention."
translated = "システムには即座の注意が必要です。"
evaluation = evaluate_translation(original, translated)
print(f"評価結果: {evaluation}")
