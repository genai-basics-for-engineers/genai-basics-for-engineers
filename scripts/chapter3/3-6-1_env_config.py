import os
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

# API キーが正しく読み込まれているか確認
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("OpenAI API キーが正しく設定されています")
else:
    print("OpenAI API キーが設定されていません")
