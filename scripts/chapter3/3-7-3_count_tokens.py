import re
import tiktoken

def optimize_input_text(text):
    """入力テキストを最適化してトークン数を削減"""
    # 余分な空白と改行を削除
    text = re.sub(r'\s+', ' ', text)

    # HTMLタグを除去
    text = re.sub(r'<[^>]+>', '', text)

    # 重複する句読点を統一
    text = re.sub(r'[。．]{2,}', '。', text)
    text = re.sub(r'[、，]{2,}', '、', text)

    # 前後の空白を削除
    text = text.strip()

    return text

def count_tokens(text, model="gpt-5-nano"):
    """指定されたモデルでのトークン数をカウント"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# 使用例
original_text = """
    これは    サンプルテキストです。。。


    不要な空白や、、、重複する句読点があります。
    <div>HTMLタグも含まれています</div>
"""

optimized_text = optimize_input_text(original_text)

print(f"元のテキスト: {count_tokens(original_text)}トークン")
print(f"最適化後: {count_tokens(optimized_text)}トークン")
print(f"削減率: {((count_tokens(original_text) - count_tokens(optimized_text)) / count_tokens(original_text) * 100):.1f}%")
