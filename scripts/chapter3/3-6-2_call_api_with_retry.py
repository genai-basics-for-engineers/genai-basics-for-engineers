import time
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
import os

load_dotenv()

# リトライ機能付きAPI呼び出し関数
def call_api_with_retry(func, retries=3, delay=1, *args, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            wait_time = min(delay * (2 ** attempt), 60)
            print(f"レート制限に達しました。{wait_time}秒後にリトライします...")
            time.sleep(wait_time)
    raise RateLimitError("リトライ回数を超過しました")

# 使用例
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_completion():
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "こんにちは"}],
    )
    return response.choices[0].message.content

# 実行
try:
    result = call_api_with_retry(create_completion)
    print(f"結果: {result}")
except RateLimitError as e:
    print(f"エラー: {e}")
