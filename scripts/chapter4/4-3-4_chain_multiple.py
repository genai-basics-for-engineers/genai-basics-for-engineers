import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY')
)
to_str = StrOutputParser()

# 要約チェーン
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは要約の専門家です。"),
    ("user",   "次の文章を一文で要約してください：\n{text}")
])
summary_chain = summary_prompt | model | to_str


# 翻訳チェーン（要約結果を受け取る）
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは翻訳の専門家です。"),
    ("user",   "次の日本語の要約を英語に翻訳してください：\n{summary}")
])

translate_chain = translate_prompt | model | to_str

# チェーンの合成：summary_chain の出力を "summary" キーで translate_chain に渡す
# データの流れ：{"text": "..."} → summary_chain実行 → {"summary": 要約} → translate_chain実行
composed_chain = {"summary": summary_chain} | translate_chain

text = """今日は来年度主力商品の新プロジェクト打ち合わせがありました。
開発期間6ヶ月でチーム全員参加し詳細計画を策定しました。
マーケティング部から顧客ニーズ、技術部から実装可能性の報告あり、
来週までに作業項目整理し次回進捗確認予定。"""

# 要約の結果を確認
summary_result = summary_chain.invoke({"text": text})
print("要約結果:", summary_result)

# 全体の結果（要約→翻訳）も確認
print("翻訳結果:", composed_chain.invoke({"text": text}))
