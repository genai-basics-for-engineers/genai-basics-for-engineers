import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 環境変数を読み込み
load_dotenv()

# LLMモデルを初期化
llm = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY')
)

# プロンプトテンプレートを作成
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なアシスタントです。回答は200文字以内でしてください。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# チェーンを作成
chain = prompt | llm

# メッセージ履歴を初期化
message_history = ChatMessageHistory()

def get_session_history(session_id: str) -> ChatMessageHistory:
    return message_history

# RunnableWithMessageHistoryでメモリ機能を追加
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 会話を実行
config = {"configurable": {"session_id": "abc123"}}

# 1回目の会話：自己紹介
print("【1回目の会話】")
input_text1 = "こんにちは！私は田中といいます。普段はエンジニアとして働いています。趣味はキャンプです。"
response1 = chain_with_history.invoke(
    {"input": input_text1},
    config=config
)
print("ユーザー: " + input_text1)
print(f"アシスタント: {response1.content}")

# 2回目の会話：週末の過ごし方の相談
print("\n【2回目の会話】")
input_text2 = "週末の過ごし方のアドバイスをください。"
response2 = chain_with_history.invoke(
    {"input": input_text2},
    config=config
)
print("ユーザー: " + input_text2)
print(f"アシスタント: {response2.content}")

# 3回目の会話：おすすめの本の相談
print("\n【3回目の会話】")
input_text3 = "おすすめの本を教えてください。"
response3 = chain_with_history.invoke(
    {"input": input_text3},
    config=config
)
print("ユーザー: " + input_text3)
print(f"アシスタント: {response3.content}")
