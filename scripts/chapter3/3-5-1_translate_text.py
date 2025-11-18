from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_text(text, target_language="Japanese"):
    """基本的な翻訳関数"""
    prompt = f"Translate the following text to {target_language}:\n{text}"

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"翻訳エラー: {e}")
        return None

# 使用例
result = translate_text("Hello, how are you today?")
print(result)
