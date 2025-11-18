#!/usr/bin/env python3
"""
改善されたSystem Promptでの技術文書レビュー
"""

import os
from openai import OpenAI

# OpenAI クライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System Promptとユーザーメッセージ
system_prompt = """あなたは20年の経験を持つシニアテクニカルライターです。以下の観点で技術文書を厳密にレビューしてください：
1. 正確性：技術的な正確性とベストプラクティスの遵守
2. 完全性：必要な情報がすべて含まれているか
3. 明確性：説明が明確で曖昧さがないか
4. 実用性：読者が実際に実装できる具体性があるか
5. セキュリティ：セキュリティ上の問題がないか

レビュー形式：良い点と改善点を明確に分け、具体的な修正案を提示し、優先度を付けて改善点を提示。"""

user_message = """文書:
# API統合ガイド
## 概要
このガイドでは、REST APIを使用してサービスと統合する方法を説明します。
## インストール
npm install api-client
## 使用方法
const client = new APIClient('your-api-key-here');
const result = client.getData();"""

print("改善されたSystem Promptでの実行結果：\n")
print("=" * 50)

# 3回実行して結果の一貫性を確認
for i in range(3):
    print(f"\n[実行 {i+1}]")
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )
    
    print(response.choices[0].message.content)
    print("\n" + "-" * 50)
