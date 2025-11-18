import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Pydanticモデルを定義
class Person(BaseModel):
    name: str = Field(description="人物の名前")
    age: int = Field(description="人物の年齢", ge=0, le=150)

parser = PydanticOutputParser(pydantic_object=Person)

# プロンプトにフォーマット指示を組み込む
prompt = ChatPromptTemplate.from_template(
    """次の文章から人物情報を抽出してください。
{format_instructions}

文章: {text}"""
).partial(format_instructions=parser.get_format_instructions())

# モデル
model = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY')
)

# Chain: プロンプト → モデル → パーサー
chain = prompt | model | parser

# 実行
result = chain.invoke({"text": "佐藤花子は30歳です。"})
print(f"名前: {result.name}, 年齢: {result.age}")
