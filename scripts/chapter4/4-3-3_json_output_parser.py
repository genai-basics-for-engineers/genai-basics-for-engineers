import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# JSON形式で返すようプロンプトを指定
prompt = ChatPromptTemplate.from_template(
"""
次の文章から人物名と年齢を抽出し、JSONで出力してください。
出力形式: {{"name": "...", "age": ...}}

文章: {text}
"""
)

# モデル
model = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY')
)

# 出力パーサー
parser = JsonOutputParser()

# Chain: プロンプト → モデル → パーサー
chain = prompt | model | parser

# 実行
result = chain.invoke({"text": "山田太郎は25歳です。"})
print(result)
