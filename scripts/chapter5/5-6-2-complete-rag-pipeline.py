#!/usr/bin/env python
"""
統合RAGパイプライン（出典付き回答）

第5章5.5節のサンプルコード
社内規定文書を使った完全なRAGシステムの実装
"""

import os
import sys
import yaml
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv

# LangChain関連のインポート
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_company_documents(data_dir: str = "company_docs") -> List[Document]:
    """YAMLフロントマター付き社内規定文書を読み込む"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"エラー: {data_dir}ディレクトリが存在しません")
        return []

    documents = []

    # .txtファイルを読み込み
    txt_files = sorted(data_path.glob("*.txt"))

    if not txt_files:
        print(f"警告: {data_dir}に文書ファイルが見つかりません")
        return []

    print(f"{len(txt_files)}個の社内規定ファイルを読み込み中...")

    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # YAMLフロントマターの解析
            if content.startswith('---'):
                parts = content.split('---', 2)
                metadata = yaml.safe_load(parts[1])
                text_content = parts[2].strip()
            else:
                metadata = {}
                text_content = content.strip()

            # デフォルトメタデータを追加
            metadata.setdefault('source', file_path.name)
            metadata.setdefault('id', file_path.stem)

            doc = Document(
                page_content=text_content,
                metadata=metadata
            )
            documents.append(doc)
        except Exception as e:
            print(f"エラー: {file_path}の読み込みに失敗: {e}")

    return documents


def format_docs(docs: List[Document]) -> str:
    """ドキュメントを出典付きでフォーマット（章節情報を含む）"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'unknown')
        reference = doc.metadata.get('reference', '')
        content = doc.page_content

        # referenceがある場合は章節情報を優先表示
        if reference:
            formatted.append(f"【出典{i+1}】{reference} ({source})\n内容: {content}")
        else:
            formatted.append(f"【出典{i+1}】{source}\n内容: {content}")
    return "\n\n".join(formatted)


def create_rag_pipeline(
    documents: Optional[List[Document]] = None,
    use_hybrid: bool = False
) -> Tuple:
    """統合RAGパイプラインの構築"""

    # ドキュメントが渡されない場合はロード
    if documents is None:
        documents = load_company_documents()
        if not documents:
            print("エラー: ドキュメントが読み込めませんでした")
            return None, None

    print(f"✓ {len(documents)}個のドキュメントを処理中...")

    # テキストの分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=160,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"✓ {len(splits)}個のチャンクに分割しました")

    # 埋め込みモデル
    embeddings = OpenAIEmbeddings()

    # Retrieverの構築
    if use_hybrid:
        print("✓ ハイブリッド検索モードを使用")

        # BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 8

        # Vector Store Retriever
        vectorstore = Chroma.from_documents(
            splits,
            embeddings,
            collection_name="rag_pipeline"
        )
        print("  - Chromaベクトルストアを使用")

        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.6, 1.0],
            c=60,
        )

        context_runnable = RunnableLambda(
            lambda question: format_docs(hybrid_retriever.invoke(question))
        )
    else:
        print("✓ ベクトル検索モードを使用")

        # Vector Store only
        vectorstore = Chroma.from_documents(
            splits,
            embeddings,
            collection_name="rag_pipeline"
        )
        print("  - Chromaベクトルストアを使用")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        context_runnable = retriever | format_docs

    # プロンプトテンプレート（出典必須・ガードレール付き）
    template = """以下の参考文書を基に質問に回答してください。

=== 参考文書 ===
{context}

=== 質問 ===
{question}

=== 回答ルール ===
1. 必ず参考文書の内容に基づいて回答してください
2. 参考文書にない情報は推測せず、「文書に記載がありません」と回答してください
3. 可能な場合は、具体的な章・節番号を含めて回答してください
   例: 「勤怠規定第3章第1節によると...」
4. 回答の最後に、参照した出典情報を【出典: 章節名】の形で明記してください
   例: 【出典: 勤怠規定 第3章第1節】

回答："""

    prompt = ChatPromptTemplate.from_template(template)

    # LLMの設定
    llm = ChatOpenAI(
        model="gpt-5-nano",  # 第5章で使用しているモデル
        temperature=0
    )

    # LCELでパイプラインを構築
    rag_chain = (
        {"context": context_runnable, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, vectorstore


def main():
    """メインの実行関数"""
    # .envファイルから環境変数を読み込み
    load_dotenv()

    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEY環境変数が設定されていません")
        sys.exit(1)

    # 社内規定文書を読み込み
    print("\n" + "=" * 60)
    print("社内規定文書を読み込み中...")
    print("=" * 60)

    documents = load_company_documents()

    if not documents:
        print("エラー: 文書が読み込めませんでした")
        return

    # RAGパイプラインの構築
    print("\n" + "=" * 60)
    print("RAGパイプラインを構築中...")
    print("=" * 60)

    # ベクトル検索のみとハイブリッド検索の両方をデモ
    for use_hybrid in [False, True]:
        mode_name = "ハイブリッド検索" if use_hybrid else "ベクトル検索"

        print(f"\n\n### {mode_name}モード ###")
        print("-" * 60)

        rag_chain, vectorstore = create_rag_pipeline(
            documents=documents,
            use_hybrid=use_hybrid
        )

        if rag_chain is None:
            continue

        # テスト質問
        test_questions = [
            "有給休暇は何日もらえますか？",
            "パスワードを忘れた場合はどうすればいいですか？",
            "経費精算の締切はいつですか？",
        ]

        for question in test_questions:
            print(f"\n質問: {question}")

            try:
                response = rag_chain.invoke(question)
                print(f"回答: {response}")
            except Exception as e:
                print(f"エラーが発生しました: {e}")

            print("-" * 40)

        # クリーンアップ
        if vectorstore:
            vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("✓ すべての処理が完了しました")
    print("=" * 60)


if __name__ == "__main__":
    main()
