from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# temperatureの違いによる出力比較（各温度で3回実行）
def compare_temperatures_multiple_runs(client, prompt, temperatures=[0.1, 0.5, 1], runs=3):
    """異なるtemperature設定で同じプロンプトを複数回実行して多様性を比較"""

    for temp in temperatures:
        print(f"\n{'='*40}")
        print(f"Temperature: {temp}")
        print(f"{'='*40}")

        for i in range(runs):

            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp # temperatureパラメータを設定
            )

            result = response.choices[0].message.content
            print(f"{i+1}: {result}")

prompt = "AIツールのキャッチコピーを1つ作成してください。"
compare_temperatures_multiple_runs(client, prompt, [0.1, 0.5, 1], 3)
