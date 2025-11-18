import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": "あなたは日本語を英語に翻訳するアシスタントです。"},
        {"role": "user", "content": "今日はどんな日ですか？"}
    ]
)

print(response.choices[0].message.content)
