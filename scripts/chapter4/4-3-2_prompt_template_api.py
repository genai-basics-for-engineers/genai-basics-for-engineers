import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": "以下の文章を英語に翻訳してください。"},
        {"role": "user", "content": "今日はいい天気ですね。"}
    ]
)

print(response.choices[0].message.content)
