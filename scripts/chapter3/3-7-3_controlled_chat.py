from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def controlled_chat(prompt, max_completion_tokens=100):
    """出力トークン数を制御したチャット"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_completion_tokens,
            temperature=0.7
        )

        usage = response.usage
        print(f"使用トークン - 入力: {usage.prompt_tokens}, 出力: {usage.completion_tokens}")

        return response.choices[0].message.content

    except Exception as e:
        print(f"エラー: {e}")
        return None

# 使用例
prompt = "Pythonプログラミングについて、200文字以内で説明してください"
result = controlled_chat(prompt, max_completion_tokens=300)
print(f"回答: {result}")
