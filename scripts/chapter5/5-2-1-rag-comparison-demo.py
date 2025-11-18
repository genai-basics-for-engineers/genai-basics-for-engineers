#!/usr/bin/env python
"""
第5章 5.1節 RAGの定義と基本構成 - RAGなし/ありの比較デモ

このスクリプトは、RAGなしとRAGありの違いを実際に体験するためのデモです。
著者が執筆時点（2025年9月）にOpenAI API (gpt-5-nano)を使用して実行・検証したものです。

実行方法:
    uv run python 5-1-1-rag-comparison-demo.py
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def demo_without_rag():
    """RAGなしの場合（失敗例）"""
    print("=" * 60)
    print("【RAGなし】LLMに直接質問する場合")
    print("=" * 60)

    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが設定されていません")
        return

    # LLMの準備（gpt-5-nano, temperature=0で固定出力）
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    # 会社固有の情報について質問
    question = "弊社の2024年度売上高を教えてください"
    print(f"\n質問: {question}")

    try:
        # LLMに直接質問
        response = llm.invoke(question)
        print(f"\n回答: {response.content}")

        # 実行結果を保存
        save_result("without_rag", question, response.content)

    except Exception as e:
        print(f"エラーが発生しました: {e}")

    print("\n→ 会社固有の情報は答えられません（当然ですが）")


def demo_with_rag():
    """RAGありの場合（成功例）"""
    print("\n" + "=" * 60)
    print("【RAGあり】検索してから回答する場合")
    print("=" * 60)

    # 会社のドキュメント（架空のデータ）
    company_docs = [
        Document(page_content="""
【2024年度 決算報告書】
売上高: 1,234億円
営業利益: 234億円
経常利益: 245億円
当期純利益: 178億円
        """),
        Document(page_content="""
【2023年度 決算報告書】
売上高: 1,156億円
営業利益: 198億円
経常利益: 205億円
当期純利益: 145億円
        """),
        Document(page_content="""
【中期経営計画 2024-2026】
2024年度目標: 売上高1,200億円（達成済み）
2025年度目標: 売上高1,400億円
2026年度目標: 売上高1,600億円
        """),
    ]

    # ベクトルストアの構築
    print("\n1. 社内文書をベクトルストアに格納中...")
    vectorstore = Chroma.from_documents(
        documents=company_docs,
        embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # RAGパイプラインの構築
    print("2. RAGパイプラインを構築中...")

    template = """あなたは社内の情報に詳しいAIアシスタントです。

参考情報：
{context}

質問： {question}

参考情報に基づいて正確に回答してください。
参考情報にない場合は「情報がありません」と答えてください。

回答："""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    # LCELでパイプライン構築
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 同じ質問をRAGで処理
    question = "弊社の2024年度売上高を教えてください"
    print(f"\n質問: {question}")

    try:
        # 検索結果を確認
        retrieved_docs = retriever.invoke(question)
        print(f"\n検索された文書（上位2件）:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"  {i}. {doc.page_content[:50]}...")

        # RAGで回答生成
        response = rag_chain.invoke(question)
        print(f"\n回答: {response}")

        # 実行結果を保存
        save_result("with_rag", question, response)

    except Exception as e:
        print(f"エラーが発生しました: {e}")

    print("\n→ 社内文書から正確な情報を取得して回答できました！")

    # クリーンアップ
    vectorstore.delete_collection()


def save_result(mode, question, answer):
    """実行結果をファイルに保存"""
    filename = f"results/5-1-1_{mode}.txt"

    # resultsディレクトリがなければ作成
    os.makedirs("results", exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"モード: {mode}\n")
        f.write(f"質問: {question}\n")
        f.write(f"回答:\n{answer}\n")

    print(f"\n結果を保存しました: {filename}")


def main():
    """メイン処理"""
    # .envファイルから環境変数を読み込み
    load_dotenv()

    print("第5章 5.1節 - RAGなし/ありの比較デモ")
    print("※このスクリプトはOpenAI APIキーが必要です")
    print("")

    # RAGなしの例
    demo_without_rag()

    # RAGありの例
    demo_with_rag()

    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    main()