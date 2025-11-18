import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# プロンプトのひな型を定義
template = "次の文章を英語に翻訳してください：\n{source_text}"

prompt = PromptTemplate(
    input_variables=["source_text"],
    template=template
)

# 入力テキスト
input_text = "今日はいい天気ですね。"

# プロンプトの表示
print('=== プロンプト ===')
print(prompt.format(source_text=input_text))

# モデルの初期化
model = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY')
)

# 出力パーサー
parser = StrOutputParser()

# Chain: プロンプト → モデル → パーサー
chain = prompt | model | parser

# 実行
print('=== LLM実行結果 ===')
result = chain.invoke({"source_text": input_text})
print(result)
