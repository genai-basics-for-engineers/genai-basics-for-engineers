import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normal_chat(prompt: str) -> str:
    """通常のチャット（ストリーミングなし）"""
    print(f"\n質問: {prompt}")
    print("回答を生成中...", end="", flush=True)

    start_time = time.time()

    try:
        # 通常のAPI呼び出し（ストリーミングなし）
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "あなたは親切なアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=3000
        )
        end_time = time.time()
        content = response.choices[0].message.content

        # 完了したら一度に表示
        print(f"\r回答 (処理時間: {end_time - start_time:.1f}秒): {content}\n")
        return content

    except Exception as e:
        print(f"\nエラー: {e}")
        return "APIエラーが発生しました。APIキーを確認してください。"

def streaming_chat(prompt: str) -> str:
    """ストリーミングチャット"""
    print(f"\n質問: {prompt}")
    print("回答: ", end="", flush=True)

    accumulated_text = ""
    start_time = time.time()

    try:
        # OpenAI APIでストリーミング
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "あなたは親切なアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            max_completion_tokens=3000
        )
        # ストリーミングレスポンスを処理
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                accumulated_text += content
                print(content, end="", flush=True)

        end_time = time.time()
        print(f" (処理時間: {end_time - start_time:.1f}秒)\n")

    except Exception as e:
        print(f"\nエラー: {e}")
        return "APIエラーが発生しました。APIキーを確認してください。"

    return accumulated_text

def main():
    """メイン実行関数"""
    print("=== OpenAI チャット比較デモ ===")
    print("コマンド:")
    print("- 'normal': 通常のチャット（完了まで待機）")
    print("- 'stream': ストリーミングチャット（リアルタイム表示）")
    print("- 'quit': 終了")

    api_key = os.getenv("OPENAI_API_KEY")
    current_mode = "stream"  # デフォルトはストリーミング

    while True:
        try:
            # モード切り替えまたは質問入力
            user_input = input(f"\n[{current_mode}モード] 質問またはコマンド: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("チャットを終了します。")
                break

            if user_input.lower() == 'normal':
                current_mode = "normal"
                print("通常モードに切り替えました。")
                continue

            if user_input.lower() == 'stream':
                current_mode = "stream"
                print("ストリーミングモードに切り替えました。")
                continue

            if not user_input:
                print("質問を入力してください。")
                continue

            # 選択されたモードでチャット実行
            if current_mode == "normal":
                normal_chat(user_input)
            else:
                streaming_chat(user_input)

        except KeyboardInterrupt:
            print("\n\nチャットを終了します。")
            break

if __name__ == "__main__":
    main()
