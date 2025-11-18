import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 基本的なAPI呼び出し
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "こんにちは"}]
)

# レスポンス構造の確認（JSON形式で出力）
print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
