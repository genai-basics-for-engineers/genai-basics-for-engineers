import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template = """
あなたは翻訳アシスタントです。
以下の文章を {source_lang} から {target_lang} に翻訳してください。

文章: {source_text}
"""

prompt = PromptTemplate(
    input_variables=["source_lang", "target_lang", "source_text"],
    template=template
)

# 入力パラメータ
input_params = {
    "source_lang": "日本語",
    "target_lang": "英語",
    "source_text": "お会いできてうれしいです。"
}

# プロンプトの表示
print('=== プロンプト ===')
print(prompt.format(**input_params))

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
result = chain.invoke(input_params)
print(result)
