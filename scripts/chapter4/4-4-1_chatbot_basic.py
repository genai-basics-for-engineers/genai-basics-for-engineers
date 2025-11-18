import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY')
)

# プロンプト（役割＋履歴＋ユーザー入力）
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは社内サポート用チャットボットです。400字以内で丁寧かつ正確に回答してください。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# チェーン（プロンプト→LLM）
chain = prompt | llm

# メッセージ履歴（セッションごとに管理）
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 会話用ラッパー
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 対話モード
def chat_mode():
    print("=== 社内サポートチャットボット ===")
    print("質問を入力してください。終了するには 'quit' または 'exit' と入力してください。\n")

    session_id = "demo-user-1"

    while True:
        try:
            # ユーザー入力を取得
            user_input = input("あなた: ").strip()

            # 終了条件
            if user_input.lower() in ['quit', 'exit', 'q', '終了']:
                print("チャットを終了します。")
                break

            # 空入力をスキップ
            if not user_input:
                continue

            # チャットボットに送信
            result = chatbot.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # 回答を表示
            print(f"ボット: {result.content}\n")

        except KeyboardInterrupt:
            print("\n\nチャットを終了します。")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}\n")

if __name__ == "__main__":
    chat_mode()
