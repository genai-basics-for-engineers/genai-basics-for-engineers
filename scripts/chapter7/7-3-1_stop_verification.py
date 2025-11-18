"""stopパラメータの動作を検証するプログラム。

このプログラムは、以下の3パターンでLLMの出力を比較します：
1. stopなし + プロンプトで「Action Inputまで停止」と明示
2. stop=["\nObservation:"]あり
3. stopなし + プロンプトで停止指示なし
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOOLS_DESCRIPTION = """
利用可能なツール一覧:
- tavily_search: Web検索を実行する

ツール詳細:
tavily_search: Web上の情報を検索して最新の情報を取得します。
  入力: 検索クエリ（文字列）
  出力: 検索結果のJSON
"""

QUESTION = "2025年のAI業界の主要な動向を調べてください"


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        sys.stderr.write(f"環境変数 {var_name} が設定されていません\n")
        sys.exit(1)
    return value


def pattern1_strong_instruction(model: str = "gpt-4o-mini") -> str:
    """パターン1: stopなし + プロンプトで明示的に停止を指示"""
    load_dotenv()
    openai_api_key = require_env("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model=model,
        temperature=0.3,  # 調査・制御系のため低めに設定
        api_key=openai_api_key,
    )

    prompt = f"""
あなたは調査を専門とするアシスタントです。以下の質問に答えるために、ツールを使ってください。

{TOOLS_DESCRIPTION}

利用方法:
```
Thought: 次に取るべきアクションを説明
Action: 使用するツール名
Action Input: ツールへの入力内容
```

**重要**: 必ず「Action Input」までで出力を停止してください。「Observation」は生成しないでください。

質問: {QUESTION}

思考過程:
"""

    response = llm.invoke(prompt)
    return response.content


def pattern2_with_stop(model: str = "gpt-4o-mini") -> str:
    """パターン2: stop=["\nObservation:"]で技術的に強制停止"""
    load_dotenv()
    openai_api_key = require_env("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model=model,
        temperature=0.3,  # 調査・制御系のため低めに設定
        api_key=openai_api_key,
        model_kwargs={"stop": ["\nObservation:"]},
    )

    prompt = f"""
あなたは調査を専門とするアシスタントです。以下の質問に答えるために、ツールを使ってください。

{TOOLS_DESCRIPTION}

利用方法:
```
Thought: 次に取るべきアクションを説明
Action: 使用するツール名
Action Input: ツールへの入力内容
```

質問: {QUESTION}

思考過程:
"""

    response = llm.invoke(prompt)
    return response.content


def pattern3_weak_instruction(model: str = "gpt-4o-mini") -> str:
    """パターン3: stopなし + 停止指示なし（弱いプロンプト）"""
    load_dotenv()
    openai_api_key = require_env("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model=model,
        temperature=0.3,  # 調査・制御系のため低めに設定
        api_key=openai_api_key,
    )

    prompt = f"""
あなたは調査を専門とするアシスタントです。以下の質問に答えるために、ツールを使ってください。

{TOOLS_DESCRIPTION}

利用方法:
```
Thought: 次に取るべきアクションを説明
Action: 使用するツール名
Action Input: ツールへの入力内容
```
ツールからの応答は Observation として返されます。

質問: {QUESTION}

思考過程:
"""

    response = llm.invoke(prompt)
    return response.content


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"

    print("=" * 80)
    print(f"stopパラメータ検証プログラム (モデル: {model})")
    print("=" * 80)
    print()

    print("【パターン1】stopなし + プロンプトで「Action Inputまで停止」と明示")
    print("-" * 80)
    try:
        result1 = pattern1_strong_instruction(model)
        print(result1)
    except Exception as e:
        print(f"エラー: {e}")
    print()
    print()

    print("【パターン2】stop=[\"\\nObservation:\"]で技術的に強制停止")
    print("-" * 80)
    try:
        result2 = pattern2_with_stop(model)
        print(result2)
    except Exception as e:
        print(f"エラー: {e}")
    print()
    print()

    print("【パターン3】stopなし + 停止指示なし（弱いプロンプト）")
    print("-" * 80)
    try:
        result3 = pattern3_weak_instruction(model)
        print(result3)
    except Exception as e:
        print(f"エラー: {e}")
    print()
    print()

    print("=" * 80)
    print("検証完了")
    print("=" * 80)
    print()
    print("【結論】")
    print("- パターン1: プロンプト指示に従うか確認")
    print("- パターン2: stopパラメータで確実に停止")
    print("- パターン3: Observationまで生成する可能性を確認")


if __name__ == "__main__":
    main()
