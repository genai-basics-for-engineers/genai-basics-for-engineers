import os
from openai import OpenAI
from dotenv import load_dotenv

# .env を読み込む
load_dotenv()

# OpenAIクライアントを初期化
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

try:
    # シンプルなテスト用プロンプト
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "user", "content": "Hello, World!"}
        ]
    )

    print("✓ API接続成功")
    print(f"レスポンス: {response.choices[0].message.content}")

except Exception as e:
    print(f"✗ API接続エラー: {str(e)}")
