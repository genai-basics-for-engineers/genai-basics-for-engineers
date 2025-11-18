from dotenv import load_dotenv
import json
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# レビュー分析の実行
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "user", "content": "以下の商品レビューを分析してJSONで返してください：'この商品は素晴らしい！配送も早くて大満足です。'"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "review_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "overall_score": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["sentiment", "confidence", "key_points", "overall_score"],
                "additionalProperties": False
            }
        }
    }
)

# 結果を取得して表示
result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2, ensure_ascii=False))
