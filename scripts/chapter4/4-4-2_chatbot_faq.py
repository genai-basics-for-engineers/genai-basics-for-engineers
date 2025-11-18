import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-nano",
    api_key=os.getenv('OPENAI_API_KEY'),
)

class FAQManager:
    def __init__(self):
        self.faq_list = []

    def add(self, question: str, answer: str):
        """FAQを追加"""
        self.faq_list.append({"question": question, "answer": answer})

    def format_for_prompt(self) -> str:
        """プロンプト用にFAQをフォーマット"""
        if not self.faq_list:
            return ""

        faq_text = "\n\n以下は事前に登録されているFAQです:\n\n"
        for i, item in enumerate(self.faq_list, 1):
            faq_text += f"Q{i}: {item['question']}\nA{i}: {item['answer']}\n\n"
        return faq_text

# FAQを登録
faq = FAQManager()
faq.add("有給休暇の申請方法", "勤怠管理システムから申請してください。直属の上司の承認が必要です。3日以上の連続休暇は1ヶ月前までに申請をお願いします。残日数は人事ポータルで確認できます。")
faq.add("交通費の精算方法", "経費精算システムから精算を行ってください。領収書（ICカードの履歴でも可）をアップロードし、訪問先・目的を記入してください。月末締めで翌月給与と一緒に振り込まれます。")

# プロンプト（役割＋FAQ＋履歴＋ユーザー入力）
system_message = f"あなたは社内サポート用チャットボットです。400字以内で丁寧かつ正確に回答してください。{faq.format_for_prompt()}ユーザーの質問がFAQに該当する場合は、FAQの情報を基に回答してください。"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
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
    print("=== 社内サポートチャットボット (FAQ対応) ===")
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
