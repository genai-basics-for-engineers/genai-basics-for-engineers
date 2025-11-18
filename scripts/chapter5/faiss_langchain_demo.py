#!/usr/bin/env python
"""
FAISSを使用したRAGデモ（LangChainラッパ使用）

第5章5.4節のサンプルコード
LangChainのFAISSラッパを使用したベクトル検索の実装例
"""

import os
import sys
from pathlib import Path
from typing import List

# LangChain関連のインポート
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_technical_documents() -> List[Document]:
    """技術文書のサンプルを作成"""
    documents = [
        """
        FAISSの基本インデックス：IndexFlatL2
        IndexFlatL2は最も基本的なインデックスで、L2距離（ユークリッド距離）を使用して
        厳密な最近傍探索を行います。全データを線形探索するため、
        小規模データでは高精度ですが、大規模データでは処理時間が増大します。
        """,
        """
        FAISSの近似インデックス：IndexHNSW
        HNSW（Hierarchical Navigable Small World）は、グラフベースの近似最近傍探索アルゴリズムです。
        階層的なグラフ構造により、高速な検索が可能で、中規模データに適しています。
        精度と速度のバランスが良いのが特徴です。
        """,
        """
        FAISSの圧縮インデックス：IndexPQ
        PQ（Product Quantization）は、ベクトルを複数のサブベクトルに分割し、
        各サブベクトルを量子化することでメモリ使用量を大幅に削減します。
        超大規模データセットでの使用に適していますが、精度は若干低下します。
        """,
        """
        LangChainでのFAISS活用
        LangChainのFAISSラッパを使用することで、複雑なインデックス管理を意識せず、
        高レベルのAPIでベクトル検索を実装できます。save_localとload_localメソッドにより、
        インデックスの永続化と再利用が簡単に行えます。
        """,
        """
        RAGシステムにおけるFAISSの位置づけ
        FAISSは高速なベクトル類似検索を提供し、RAGシステムの検索部分を担います。
        ChromaやPineconeと比較して、より細かい制御が可能で、
        大規模データセットでのパフォーマンス最適化に優れています。
        """
    ]

    return [
        Document(
            page_content=text.strip(),
            metadata={"source": f"faiss_doc_{i+1}.md", "id": i+1, "topic": "FAISS"}
        )
        for i, text in enumerate(documents)
    ]


def demonstrate_faiss_operations():
    """FAISSの基本操作のデモンストレーション"""

    print("=== FAISSベクトルストアのデモ ===\n")

    # サンプル文書の作成
    documents = create_technical_documents()
    print(f"サンプル文書を{len(documents)}件作成しました。")

    # テキストの分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    # 埋め込みモデルの初期化
    embeddings = OpenAIEmbeddings()

    # FAISSベクトルストアの作成
    print("\n1. FAISSベクトルストアを作成中...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    print(f"   → {len(splits)}個のドキュメントをインデックスに追加しました。")

    # インデックスの保存
    index_path = "faiss_index"
    print(f"\n2. インデックスを '{index_path}' に保存中...")
    vectorstore.save_local(index_path)
    print(f"   → インデックスを保存しました。")

    # インデックスの読み込み
    print(f"\n3. 保存したインデックスを読み込み中...")
    loaded_vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True  # 信頼できるソースからのデータのみ許可
    )
    print(f"   → インデックスを読み込みました。")

    # 類似検索のデモ
    queries = [
        "FAISSの高速化技術について",
        "大規模データでの最適化手法",
        "LangChainとの統合方法"
    ]

    print("\n4. 類似検索のデモ:")
    for query in queries:
        print(f"\n   質問: {query}")
        print("   " + "-" * 50)

        # 類似検索（上位3件）
        results = loaded_vectorstore.similarity_search(query, k=3)

        for i, doc in enumerate(results):
            source = doc.metadata.get('source', 'unknown')
            content = doc.page_content[:150] + "..."
            print(f"   【結果{i+1}】{source}")
            print(f"   内容: {content}")

    # スコア付き類似検索
    print("\n5. スコア付き類似検索のデモ:")
    query = "IndexPQの特徴"
    print(f"   質問: {query}")
    print("   " + "-" * 50)

    results_with_scores = loaded_vectorstore.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results_with_scores):
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content[:100] + "..."
        print(f"   【結果{i+1}】{source} (距離スコア: {score:.4f})")
        print(f"   内容: {content}")

    # 最大限界検索（MMR: Maximal Marginal Relevance）
    print("\n6. MMR（多様性を考慮した検索）のデモ:")
    query = "FAISSのインデックス"
    print(f"   質問: {query}")
    print("   " + "-" * 50)

    mmr_results = loaded_vectorstore.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=10  # 初期候補として10件取得
    )

    for i, doc in enumerate(mmr_results):
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content[:100] + "..."
        print(f"   【結果{i+1}】{source}")
        print(f"   内容: {content}")

    # インデックスファイルのクリーンアップ
    import shutil
    if Path(index_path).exists():
        shutil.rmtree(index_path)
        print(f"\n→ インデックスファイルを削除しました。")


def main():
    """メインの実行関数"""

    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEY環境変数が設定されていません")
        sys.exit(1)

    try:
        # FAISSのインポート確認
        import faiss
        print(f"FAISS バージョン: {faiss.__version__}")
    except ImportError:
        print("警告: FAISSがインストールされていません。")
        print("以下のコマンドでインストールしてください:")
        print("  pip install faiss-cpu  # CPU版")
        print("  pip install faiss-gpu  # GPU版（CUDA環境必須）")
        sys.exit(1)

    # デモの実行
    demonstrate_faiss_operations()


if __name__ == "__main__":
    main()