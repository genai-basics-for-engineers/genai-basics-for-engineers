#!/usr/bin/env python
"""
第5章 5.3節 検索の基礎：キーワード検索 vs ベクトル検索 - 比較デモ

このスクリプトは、キーワード検索とベクトル検索の違いを実際に体験するためのデモです。
著者が執筆時点（2025年9月）にOpenAI API (gpt-5-nano)を使用して実行・検証したものです。

実行方法:
    uv run python 5-3-1-search-comparison.py
"""

import os
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from typing import List, Tuple
from pathlib import Path
from dotenv import load_dotenv


def prepare_documents() -> Tuple[List[str], str]:
    """サンプル文書とクエリを準備"""
    documents = [
        "パスワードを忘れた場合は、ログイン画面の「パスワードを忘れた方」をクリックしてください。",
        "アカウントがロックされた際は、30分待つか管理者に連絡してください。",
        "初回ログイン時は、仮パスワードから新しいパスワードへの変更が必要です。"
    ]

    query = "パスワードリセットの方法"

    return documents, query


def simple_tokenizer(text: str) -> List[str]:
    """簡単な日本語トークナイザー（単純な文字種で分割）"""
    # 実際は形態素解析を使うべきですが、デモでは簡略化
    import re

    tokens = []

    # 主要なキーワードを抽出
    # カタカナ語（2文字以上）
    tokens.extend(re.findall(r'[ァ-ヴー]{2,}', text))
    # 漢字の連続（2文字以上）
    tokens.extend(re.findall(r'[一-龠]{2,}', text))
    # ひらがなの連続（3文字以上の意味のある単語）
    tokens.extend(re.findall(r'[ぁ-ん]{3,}', text))
    # 英数字
    tokens.extend(re.findall(r'[a-zA-Z0-9]+', text))

    # 重要なキーワードを個別に追加（完全一致を重視）
    keywords = ['パスワード', 'リセット', '方法', '忘れた', 'ログイン',
                'アカウント', 'ロック', '変更', '初回', '管理者']
    for keyword in keywords:
        if keyword in text:
            tokens.append(keyword)

    return tokens


def save_result(filename: str, content: str):
    """結果をファイルに保存"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n結果を保存しました: {filepath}")


def demo_keyword_search(documents: List[str], query: str):
    """キーワード検索（BM25）のデモ"""
    print("=" * 60)
    print("【キーワード検索（BM25）】")
    print("=" * 60)

    print(f"\n質問: {query}")
    print("\n文書:")
    for i, doc in enumerate(documents, 1):
        print(f"  文書{i}: {doc[:30]}...")

    # 単語に分割（簡略版）
    tokenized_docs = [simple_tokenizer(doc) for doc in documents]
    tokenized_query = simple_tokenizer(query)

    print(f"\nクエリのトークン: {tokenized_query}")

    # BM25でスコア計算
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    print("\nキーワード検索の結果:")
    for i, score in enumerate(scores, 1):
        print(f"  文書{i}: スコア {score:.2f}")

    # 最高スコアの文書を表示
    best_idx = np.argmax(scores)
    print(f"\n最も関連する文書: 文書{best_idx + 1}")
    print(f"内容: {documents[best_idx][:50]}...")

    # 結果を保存
    output = []
    output.append("キーワード検索（BM25）の結果")
    output.append("=" * 40)
    output.append(f"\n質問: {query}")
    output.append(f"\nクエリのトークン: {tokenized_query}")
    output.append("\n各文書のスコア:")
    for i, score in enumerate(scores, 1):
        output.append(f"  文書{i}: スコア {score:.2f}")
    output.append(f"\n最も関連する文書: 文書{best_idx + 1}")
    output.append(f"内容: {documents[best_idx]}")

    save_result("5-3-1_keyword_search.txt", "\n".join(output))


def demo_vector_search(documents: List[str], query: str):
    """ベクトル検索のデモ"""
    print("\n" + "=" * 60)
    print("【ベクトル検索（埋め込み）】")
    print("=" * 60)

    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが設定されていません")
        return

    print(f"\n質問: {query}")

    try:
        embeddings = OpenAIEmbeddings()

        print("\n1. 文書とクエリをベクトル化中...")
        # 文書とクエリを数値ベクトルに変換
        doc_vectors = embeddings.embed_documents(documents)
        query_vector = embeddings.embed_query(query)

        print(f"   ベクトルの次元数: {len(doc_vectors[0])}次元")

        print("\n2. コサイン類似度を計算中...")
        # コサイン類似度を計算
        similarities = []
        for doc_vec in doc_vectors:
            similarity = np.dot(query_vector, doc_vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vec)
            )
            similarities.append(similarity)

        print("\nベクトル検索の結果:")
        for i, sim in enumerate(similarities, 1):
            print(f"  文書{i}: 類似度 {sim:.2f}")

        # 最高類似度の文書を表示
        best_idx = np.argmax(similarities)
        print(f"\n最も関連する文書: 文書{best_idx + 1}")
        print(f"内容: {documents[best_idx][:50]}...")

        # 結果を保存
        output = []
        output.append("ベクトル検索（埋め込み）の結果")
        output.append("=" * 40)
        output.append(f"\n質問: {query}")
        output.append(f"\nベクトルの次元数: {len(doc_vectors[0])}次元")
        output.append("\n各文書の類似度:")
        for i, sim in enumerate(similarities, 1):
            output.append(f"  文書{i}: 類似度 {sim:.2f}")
        output.append(f"\n最も関連する文書: 文書{best_idx + 1}")
        output.append(f"内容: {documents[best_idx]}")

        save_result("5-3-1_vector_search.txt", "\n".join(output))

    except Exception as e:
        print(f"エラーが発生しました: {e}")

def main():
    """メイン処理"""
    # .envファイルから環境変数を読み込み
    load_dotenv()

    print("第5章 5.3節 - キーワード検索 vs ベクトル検索の比較デモ")
    print("※ベクトル検索はOpenAI APIキーが必要です")
    print("")

    # サンプル文書とクエリの準備
    documents, query = prepare_documents()

    # キーワード検索のデモ
    demo_keyword_search(documents, query)

    # ベクトル検索のデモ
    demo_vector_search(documents, query)

    # まとめを保存
    summary = []
    summary.append("検索手法の比較まとめ")
    summary.append("=" * 40)
    summary.append("\n• キーワード検索: 文字の完全一致を重視（エラーコードなどに最適）")
    summary.append("• ベクトル検索: 意味の類似性を重視（自然言語の質問に最適）")
    summary.append("• ハイブリッド検索: 両方のいいとこ取り（実用的な選択）")

    save_result("5-3-1_summary.txt", "\n".join(summary))

    print("\n" + "=" * 60)
    print("全ての結果をresults/ディレクトリに保存しました")
    print("=" * 60)


if __name__ == "__main__":
    main()
