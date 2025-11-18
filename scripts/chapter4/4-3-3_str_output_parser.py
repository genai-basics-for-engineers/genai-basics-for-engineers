import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# プロンプト
prompt = ChatPromptTemplate.from_template(
    "次の文章を100文字以内で要約してください:\n{text}"
)

# モデル
model = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY')
)

target_text = """
LangChainは、大規模言語モデル（LLM）を活用したアプリケーション開発を支援するフレームワークです。
プロンプト設計や会話履歴の管理、外部データ検索、APIとの連携などをモジュール化して提供し、
必要な機能を組み合わせることで柔軟にアプリを構築できます。
チャットボットやRAGといった実用システム開発を効率化します。
"""

print("【StrOutputParserを使用しない場合】")

# Chain: プロンプト → モデル（パーサーなし）
chain_without_parser = prompt | model

# 実行
result_without_parser = chain_without_parser.invoke({"text": target_text})
print(f"結果: {result_without_parser}")

print("\n【StrOutputParserを使用した場合】")
# 出力パーサー
parser = StrOutputParser()

# Chain: プロンプト → モデル → パーサー
chain_with_parser = prompt | model | parser

# 実行
result_with_parser = chain_with_parser.invoke({"text": target_text})
print(f"結果: {result_with_parser}")
