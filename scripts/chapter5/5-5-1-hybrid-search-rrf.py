#!/usr/bin/env python
"""
ハイブリッド検索デモ（BM25＋ベクトル＋RRF）

第5章5.5節のサンプルコード
LangChainのEnsembleRetrieverを使用したハイブリッド検索の実装
"""

import os
import sys
import re
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# LangChain関連のインポート
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_documents_from_files() -> List[Document]:
    """hybrid_sample_data/ディレクトリからドキュメントを読み込み"""
    current_dir = Path(__file__).parent
    data_dir = current_dir / "hybrid_sample_data"

    if not data_dir.exists():
        raise FileNotFoundError(f"データディレクトリが見つかりません: {data_dir}")

    documents = []

    # .txtファイルを番号順にソート
    txt_files = sorted(data_dir.glob("hybrid_doc_*.txt"))

    for i, file_path in enumerate(txt_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            document = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "id": i + 1,
                    "file_path": str(file_path)
                }
            )
            documents.append(document)

        except Exception as e:
            print(f"ファイル読み込みエラー {file_path}: {e}")
            continue

    if not documents:
        raise ValueError("読み込めるドキュメントファイルがありません")

    return documents


def tokenize_for_bm25(text: str) -> list[str]:
    """
    日本語を含むテキストを軽量にトークナイズする。
    - 英数字はそのまま1トークン。
    - 漢字・ひらがな・カタカナの連続部分は文字bi-gramに分解し、部分一致しやすくする。
    """
    tokens: list[str] = []
    for chunk in re.findall(r"[A-Za-z0-9]+|[一-龥ぁ-んァ-ンー]+", text):
        if re.fullmatch(r"[A-Za-z0-9]+", chunk):
            tokens.append(chunk)
        else:
            # 日本語部分は bi-gram で分割して BM25 のヒット率を上げる
            if len(chunk) == 1:
                tokens.append(chunk)
            else:
                tokens.extend(chunk[i: i + 2] for i in range(len(chunk) - 1))
    return tokens


def save_result(filename: str, content: str):
    """結果をファイルに保存"""
    from pathlib import Path
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / filename, 'w', encoding='utf-8') as f:
        f.write(content)


def compare_retrieval_methods(splits: List[Document]):
    """検索手法の比較デモ"""

    output = []
    output.append("\n=== 検索手法の比較 ===\n")
    print(output[-1])

    # 1. BM25 Retrieverの構築
    bm25_retriever = BM25Retriever.from_documents(
        splits,
        preprocess_func=tokenize_for_bm25,
    )
    bm25_retriever.k = 3

    # 2. ベクトル検索 Retrieverの構築
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="hybrid_demo"
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # テストクエリ（本文と同じ2件）
    queries = [
        "X-Pack monitoring の設定方法",                    # 固有名詞・製品名でBM25が有利
        "セマンティック検索のメリット",                       # 抽象表現でベクトルが有利
    ]

    for query in queries:
        output.append(f"クエリ: '{query}'")
        output.append("=" * 60)
        print(f"クエリ: '{query}'")
        print("=" * 60)

        # BM25検索
        output.append("\n【BM25検索結果】")
        print("\n【BM25検索結果】")
        bm25_results = bm25_retriever.invoke(query)
        for i, doc in enumerate(bm25_results):
            title = doc.page_content.splitlines()[0]
            output.append(f"  {i+1}. {doc.metadata['source']}: {title}")
            print(f"  {i+1}. {doc.metadata['source']}: {title}")

        # ベクトル検索
        output.append("\n【ベクトル検索結果】")
        print("\n【ベクトル検索結果】")
        vector_results = dense_retriever.invoke(query)
        for i, doc in enumerate(vector_results):
            title = doc.page_content.splitlines()[0]
            output.append(f"  {i+1}. {doc.metadata['source']}: {title}")
            print(f"  {i+1}. {doc.metadata['source']}: {title}")

        output.append("\n" + "-" * 60 + "\n")
        print("\n" + "-" * 60 + "\n")

    save_result("5-4-1-hybrid-search-comparison.txt", "\n".join(output))

    # クリーンアップ
    vectorstore.delete_collection()


def demonstrate_ensemble_retriever(splits: List[Document]):
    """EnsembleRetrieverのデモンストレーション"""

    output = []
    output.append("\n=== EnsembleRetriever（ハイブリッド検索）のデモ ===\n")
    print(output[-1])

    # BM25 Retrieverの構築
    bm25_retriever = BM25Retriever.from_documents(
        splits,
        preprocess_func=tokenize_for_bm25,
    )
    bm25_retriever.k = 12  # 多めに候補を取得

    # ベクトル検索 Retrieverの構築
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="ensemble_demo"
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # EnsembleRetrieverでハイブリッド検索を構築（内部でRRF）
    # weights: 各Retrieverの重み付け（合計1でなくてよい）
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.6, 1.0],  # BM25を少し抑えめ、ベクトルを重視
        c=60,               # RRFのk（高順位をどれだけ強調するか）
    )

    # テストクエリ（ハイブリッドデモ用）
    test_queries = [
        "X-Pack monitoring の設定方法",
        "セマンティック検索のメリット",
    ]

    for query in test_queries:
        output.append(f"クエリ: '{query}'")
        output.append("-" * 50)
        print(f"クエリ: '{query}'")
        print("-" * 50)

        # ハイブリッド検索の実行
        results = hybrid_retriever.invoke(query)

        # 上位4件を表示（ファイル名＋タイトル）
        output.append("ハイブリッド検索結果（上位4件）:")
        print("ハイブリッド検索結果（上位4件）:")
        for i, doc in enumerate(results[:4]):
            source = doc.metadata.get('source', 'unknown')
            title = doc.page_content.splitlines()[0]
            output.append(f"  【結果{i+1}】{source}: {title}")
            print(f"  【結果{i+1}】{source}: {title}")

        output.append("")
        print()

    save_result("5-4-1-ensemble-results.txt", "\n".join(output))

    # 重み付けの影響を確認（1クエリで1位が入れ替わる例）
    output2 = []
    output2.append("\n=== 重み付けの影響の確認 ===\n")
    print(output2[-1])

    weight_configs = [
        (0.9, 0.1, "BM25重視"),
        (0.5, 0.5, "均等"),
        (0.1, 0.9, "ベクトル重視"),
    ]

    query = "セマンティック検索のメリット"  # 重みで1位が揺れやすいクエリ

    for bm25_w, vec_w, desc in weight_configs:
        output2.append(f"\n設定: {desc} (BM25={bm25_w}, Vector={vec_w})")
        output2.append("-" * 40)
        print(f"\n設定: {desc} (BM25={bm25_w}, Vector={vec_w})")
        print("-" * 40)

        # 異なる重み付けでEnsembleRetrieverを構築
        weighted_hybrid = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[bm25_w, vec_w],
            c=60,
        )

        results = weighted_hybrid.invoke(query)
        output2.append("上位3件の結果:")
        print("上位3件の結果:")
        for i, doc in enumerate(results[:3]):
            output2.append(f"  {i+1}. {doc.metadata['source']}")
            print(f"  {i+1}. {doc.metadata['source']}")

    save_result("5-4-1-weight-comparison.txt", "\n".join(output2))

    # クリーンアップ
    vectorstore.delete_collection()


def main():
    """メインの実行関数"""
    # .envファイルから環境変数を読み込み
    load_dotenv()

    # サンプル文書の読み込み
    documents = load_documents_from_files()
    print(f"サンプル文書を{len(documents)}件読み込みました。")

    # テキストの分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=160,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"文書を{len(splits)}個のチャンクに分割しました。")

    # 検索手法の比較
    compare_retrieval_methods(splits)

    # EnsembleRetrieverのデモ
    demonstrate_ensemble_retriever(splits)

    print("\n✓ ハイブリッド検索デモが完了しました。")


if __name__ == "__main__":
    main()
