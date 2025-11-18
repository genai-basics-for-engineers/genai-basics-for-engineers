#!/usr/bin/env python
"""
最小RAGシステムのデモ（LangChain/LCEL使用）

第5章5.3節のサンプルコード
LCELを使用してChromaとOpenAIを組み合わせた最小構成のRAGシステム
"""

import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# LangChain関連のインポート
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def load_faq_documents(data_dir: str = "sample_data") -> List[Document]:
    """FAQドキュメントを読み込む"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"エラー: {data_dir}ディレクトリが存在しません")
        return []

    documents = []

    # FAQファイルを読み込み
    faq_files = sorted(data_path.glob("faq_*.txt"))

    if not faq_files:
        print(f"警告: {data_dir}にFAQファイルが見つかりません")
        return []

    print(f"{len(faq_files)}個のFAQファイルを読み込み中...")

    for file_path in faq_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "id": file_path.stem
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"エラー: {file_path}の読み込みに失敗: {e}")

    return documents


def format_docs(docs: List[Document]) -> str:
    """検索結果を出典付きでフォーマット"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content
        formatted.append(f"【出典{i+1}】{source}\n内容: {content}")
    return "\n\n".join(formatted)


def demo_vector_search():
    """ベクトル検索のデモ"""
    print("=" * 60)
    print("ベクトル検索デモ")
    print("=" * 60)

    # FAQドキュメントを読み込み
    documents = load_faq_documents()

    if not documents:
        print("ドキュメントが読み込めませんでした")
        return None

    print(f"\n{len(documents)}個のドキュメントを読み込みました")

    # 埋め込みとベクトルストアの作成
    print("\nベクトルストアを作成中...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="faq_demo"
    )

    print("ベクトルストアの作成が完了しました")

    # ベクトル検索のテスト
    print("\n" + "=" * 60)
    print("検索テスト")
    print("=" * 60)

    query = "VPN繋がらない"
    print(f"\n検索クエリ: {query}")
    results = vectorstore.similarity_search(query, k=3)

    print("\n検索結果（上位3件）:")
    for i, doc in enumerate(results):
        print(f"\n結果{i+1}: {doc.metadata['source']}")
        print(f"内容: {doc.page_content[:100]}...")

    return vectorstore


def create_rag_pipeline(vectorstore=None):
    """RAGパイプラインの構築"""

    if vectorstore is None:
        # ベクトルストアがない場合は作成
        documents = load_faq_documents()
        if not documents:
            print("エラー: ドキュメントが読み込めませんでした")
            return None

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="faq_demo"
        )

    # Retrieverの作成（上位4件を取得）
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # プロンプトテンプレート
    prompt = ChatPromptTemplate.from_template("""
以下の文脈を使って質問に答えてください。
文脈に答えがない場合は、「提供された情報では回答できません」と答えてください。
必ず文脈に基づいて回答し、推測は避けてください。

文脈:
{context}

質問: {question}

回答:""")

    # LLMの初期化（gpt-5-nanoを使用）
    llm = ChatOpenAI(
        model="gpt-5-nano",  # 実際の利用可能なモデル
        temperature=0
    )

    # LCELパイプラインの構築
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    """メイン処理"""
    # .envファイルから環境変数を読み込み
    load_dotenv()

    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEY環境変数が設定されていません")
        sys.exit(1)

    # ベクトル検索のデモ
    vectorstore = demo_vector_search()

    if vectorstore is None:
        return

    # RAGパイプラインの構築と実行
    print("\n" + "=" * 60)
    print("RAGパイプライン実行")
    print("=" * 60)

    rag_chain = create_rag_pipeline(vectorstore)

    if rag_chain is None:
        return

    # テスト質問
    test_questions = [
        "VPNの接続方法を教えてください",
        "経費精算の締切はいつですか？",
        "パスワードを忘れた場合はどうすればいいですか？"
    ]

    for question in test_questions:
        print(f"\n質問: {question}")
        try:
            answer = rag_chain.invoke(question)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
        print("-" * 40)

    # インタラクティブモード
    print("\n" + "=" * 60)
    print("インタラクティブモード（終了は'exit'または'quit'）")
    print("=" * 60)

    while True:
        user_question = input("\n質問を入力してください: ")
        if user_question.lower() in ['exit', 'quit', 'q']:
            print("終了します")
            break

        try:
            answer = rag_chain.invoke(user_question)
            print(f"\n回答: {answer}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()