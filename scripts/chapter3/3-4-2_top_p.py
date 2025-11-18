from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# top_pの違いによる出力比較（各top_pで3回実行）
def compare_top_p_multiple_runs(client, prompt, top_p=[0.1, 0.5, 1], runs=3):
    """異なるtop_p設定で同じプロンプトを複数回実行して多様性を比較"""

    for p in top_p:
        print(f"\n{'='*40}")
        print(f"top_p: {p}")
        print(f"{'='*40}")

        for i in range(runs):
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                top_p=p
            )

            result = response.choices[0].message.content
            print(f"{i+1}: {result}")

prompt = "冬をテーマにした俳句を1つ作成して1行で表示してください。"
compare_top_p_multiple_runs(client, prompt, [0.1, 0.5, 1], 3)
