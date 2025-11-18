#!/usr/bin/env python
"""
第5章 5.3節 ハイブリッド検索の簡単なデモ

キーワード検索とベクトル検索を組み合わせた簡易ハイブリッド検索の実装。
RRF（Reciprocal Rank Fusion）を使用して結果を統合します。

実行方法:
    uv run python 5-2-2-simple-hybrid-search.py
"""

import os
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Tuple
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


def keyword_search(query: str, documents: List[str]) -> List[Tuple[int, float]]:
    """キーワード検索（BM25）を実行"""
    # 単語に分割
    tokenized_docs = [simple_tokenizer(doc) for doc in documents]
    tokenized_query = simple_tokenizer(query)

    # BM25でスコア計算
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    # (文書インデックス, スコア)のリストを返す
    results = [(i, score) for i, score in enumerate(scores)]
    # スコアの降順でソート
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def vector_search(query: str, documents: List[str]) -> List[Tuple[int, float]]:
    """ベクトル検索を実行"""
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: OPENAI_API_KEYが設定されていません（疑似結果を返します）")
        # 疑似結果を返す
        return [(0, 0.90), (2, 0.89), (1, 0.84)]

    try:
        embeddings = OpenAIEmbeddings()

        # 文書とクエリを数値ベクトルに変換
        doc_vectors = embeddings.embed_documents(documents)
        query_vector = embeddings.embed_query(query)

        # コサイン類似度を計算
        similarities = []
        for doc_vec in doc_vectors:
            similarity = np.dot(query_vector, doc_vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vec)
            )
            similarities.append(similarity)

        # (文書インデックス, 類似度)のリストを返す
        results = [(i, sim) for i, sim in enumerate(similarities)]
        # 類似度の降順でソート
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    except Exception as e:
        print(f"エラー: {e}")
        return []


def hybrid_search(query: str, documents: List[str]) -> List[Tuple[int, float]]:
    """ハイブリッド検索（簡略版）"""
    # キーワード検索の結果
    keyword_results = keyword_search(query, documents)
    print("\nキーワード検索の結果（順位）:")
    for rank, (doc_idx, score) in enumerate(keyword_results, 1):
        print(f"  {rank}位: 文書{doc_idx + 1}（スコア: {score:.2f}）")

    # ベクトル検索の結果
    vector_results = vector_search(query, documents)
    print("\nベクトル検索の結果（順位）:")
    for rank, (doc_idx, similarity) in enumerate(vector_results, 1):
        print(f"  {rank}位: 文書{doc_idx + 1}（類似度: {similarity:.2f}）")

    # RRF（順位の逆数で得点化）で統合
    scores = {}
    k = 60  # RRFのパラメータ（通常60を使用）

    # キーワード検索結果のスコアを加算
    for rank, (doc_idx, _) in enumerate(keyword_results):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1/(k + rank + 1)

    # ベクトル検索結果のスコアを加算
    for rank, (doc_idx, _) in enumerate(vector_results):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1/(k + rank + 1)

    # スコアでソート
    final_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return final_results


def save_result(filename: str, content: str):
    """結果をファイルに保存"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n結果を保存しました: {filepath}")


def main():
    """メイン処理"""
    # .envファイルから環境変数を読み込み
    load_dotenv()

    print("第5章 5.2節 - ハイブリッド検索の簡単なデモ")
    print("=" * 60)

    # サンプル文書とクエリの準備
    documents, query = prepare_documents()

    print(f"\n質問: {query}")
    print("\n文書:")
    for i, doc in enumerate(documents, 1):
        print(f"  文書{i}: {doc[:30]}...")

    print("\n" + "=" * 60)
    print("【ハイブリッド検索の実行】")
    print("=" * 60)

    # ハイブリッド検索を実行
    hybrid_results = hybrid_search(query, documents)

    print("\n" + "=" * 60)
    print("【最終結果（RRFで統合）】")
    print("=" * 60)

    output = []
    output.append("ハイブリッド検索（RRF統合）の結果")
    output.append("=" * 40)
    output.append(f"\n質問: {query}")
    output.append("\n最終ランキング:")

    for rank, (doc_idx, rrf_score) in enumerate(hybrid_results, 1):
        result_line = f"  {rank}位: 文書{doc_idx + 1}（RRFスコア: {rrf_score:.4f}）"
        print(result_line)
        output.append(result_line)

        # 1位の文書内容を表示
        if rank == 1:
            content_line = f"       内容: {documents[doc_idx][:50]}..."
            print(content_line)
            output.append(content_line)

    # 結果を保存
    save_result("5-2-2_hybrid_search.txt", "\n".join(output))

if __name__ == "__main__":
    main()
