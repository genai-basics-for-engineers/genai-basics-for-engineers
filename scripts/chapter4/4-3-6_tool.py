import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

# ツール定義
def multiply(a: int, b: int) -> int:
    """二つの整数を掛け算します"""
    return a * b

tools = [multiply]
llm = ChatOpenAI(model="gpt-5-nano", api_key=os.getenv("OPENAI_API_KEY"))

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="与えられたメッセージに従って計算をしてください"
)

result = agent.invoke({"messages": [{"role": "user", "content": "7に8を掛けてください"}]})

# 結果は最後のメッセージのcontentに含まれる
print(result["messages"][-1].content)
