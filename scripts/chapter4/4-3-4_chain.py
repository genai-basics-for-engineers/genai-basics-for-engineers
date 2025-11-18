import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# プロンプト
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは日本語を英語に翻訳するアシスタントです。"),
    ("user", "次の文章を英語に翻訳してください：\n{text}")
])

# モデル
model = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY')
)

# 出力パーサー
parser = StrOutputParser()

# Chain: プロンプト → モデル → パーサー
chain = prompt | model | parser

# 実行
result = chain.invoke({"text": "今日はどんな日ですか？"})
print(result)
