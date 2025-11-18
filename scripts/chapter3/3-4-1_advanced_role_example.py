from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_system_role(system_content, user_content):
    """システムロールを使用してAIの振る舞いを設定"""
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

# 技術専門家として
print("技術専門家として:")
tech_response = chat_with_system_role(
    "あなたは技術的な質問に正確に答える専門家です。",
    "LLMとは何ですか？200文字以内で説明してください。"
)
if tech_response:
    print(tech_response)

# 初心者向けアシスタントとして
print("\n初心者向けアシスタントとして:")
beginner_response = chat_with_system_role(
    "あなたは初心者にも分かりやすく説明する親切なアシスタントです。",
    "LLMとは何ですか？200文字以内で説明してください。"
)
if beginner_response:
    print(beginner_response)
