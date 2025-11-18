import tiktoken
import re

def split_text(text, model="gpt-5-nano", max_completion_tokens=3000):
    """テキストをトークン数に応じて段落・文単位で分割"""
    # モデルに対応するエンコーディングを取得（なければ汎用型を使用）
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    count_tokens = lambda s: len(encoding.encode(s))
    chunks = []

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # 段落が長すぎる場合は文単位で分割
        if count_tokens(paragraph) > max_completion_tokens:
            sentences = re.split(r'(?<=[。！？.!?])\s*', paragraph)
            current = ""
            for sentence in sentences:
                if not sentence.strip():
                    continue
                if count_tokens(current + sentence) > max_completion_tokens:
                    if current:
                        chunks.append(current.strip())
                    current = sentence
                else:
                    current += sentence
            if current:
                chunks.append(current.strip())
        else:
            chunks.append(paragraph)

    return chunks

# テスト用サンプル
if __name__ == "__main__":
    sample_text = """これは最初の段落です。この段落では日本語のテキスト分割について説明します。

これは二番目の段落です。この段落は長い文章を含んでいます。文章が長すぎる場合は、適切に分割される必要があります。トークン数を考慮して分割します。

これは非常に長い段落のサンプルです。この段落には多くの文が含まれています。GPT-5のトークン制限を考慮して、適切なサイズにチャンクを分割する必要があります。"""

    chunks = split_text(sample_text, max_completion_tokens=100)
    print(f"テキストが {len(chunks)} 個のチャンクに分割されました：\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"チャンク {i}: {chunk}")
