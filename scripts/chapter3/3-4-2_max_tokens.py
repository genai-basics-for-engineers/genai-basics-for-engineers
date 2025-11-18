from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

def adaptive_token_chat(prompt: str, max_completion_tokens: int = 300):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_completion_tokens,
        temperature=0.7
    )

    return response.choices[0].message.content

# 使用例
if __name__ == "__main__":
    detailed_prompt = "機械学習における深層学習の役割と主要なアプリケーション領域について説明してください"
    print(f"質問: {detailed_prompt}")
    answer = adaptive_token_chat(detailed_prompt, 300)
    print(f"回答: {answer}")
