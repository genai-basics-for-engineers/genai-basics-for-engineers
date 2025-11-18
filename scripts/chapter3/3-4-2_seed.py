from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

def consistent_generation(prompt, seed_value=12345):
    """
    同じ入力に対して毎回同じ出力を得る関数

    Args:
        prompt (str): 入力プロンプト
        seed_value (int): 再現性確保のためのシード値

    Returns:
        str: AI生成のレスポンス
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        seed=seed_value,
        temperature=0.7
    )

    return response.choices[0].message.content

# 使用例
if __name__ == "__main__":

    prompt = "短い詩を1つ書いてください"

    # 同じシード値で複数回実行すると同じ結果が得られる
    result1 = consistent_generation(prompt)
    result2 = consistent_generation(prompt)

    print("1回目の結果:")
    print(result1)
    print("\n2回目の結果:")
    print(result2)
