from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_explanation_prompt(user_role, user_level="beginner", language="ja"):
    """対象者・理解度・言語に応じた説明プロンプトを生成"""
    template = """
あなたは教育の専門家です。

次の内容を{language}で300文字以内で説明してください。
対象者: {role}
理解度レベル: {level}

内容:
{topic}

説明の条件:
- 営業向けなら、ビジネス的な価値や活用例を重視
- エンジニア向けなら、技術的な詳細や仕組みを重視
- 初心者には専門用語を避け、具体例を交える
- 上級者には専門的な背景知識や用語も含める
"""
    role = "営業担当者" if user_role == "sales" else "エンジニア"
    level = "初心者" if user_level == "beginner" else "上級者"
    lang = "日本語" if language == "ja" else "English"

    return template.format(language=lang, role=role, level=level, topic="機械学習の基礎")

def run_example(user_role, user_level, language):
    """プロンプトを生成し、モデルに投げて結果を表示"""
    prompt = build_explanation_prompt(user_role, user_level, language)
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
    )

    print("=== プロンプト ===")
    print(prompt)
    print("=== 出力結果 ===")
    print(response.choices[0].message.content.strip())

# パターン1: 営業担当者向け・初心者・日本語
run_example(user_role="sales", user_level="beginner", language="ja")

# パターン2: エンジニア向け・上級者・日本語
run_example(user_role="engineer", user_level="advanced", language="en")
