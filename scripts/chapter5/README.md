# 第5章 RAGを実装してみよう - サンプルコード

このディレクトリには、書籍「エンジニアのための生成AI入門ガイドブック」第5章のサンプルコードが含まれています。

## 概要

第5章では、RAG（Retrieval-Augmented Generation）システムをLangChain/LCELを使用して実装します。決定的パス（非Agentic）のRAG実装に焦点を当て、出典付き回答の生成まで一連の流れを学習できます。

## 🚀 uvを使った実行方法（推奨）

本書のサンプルコードは、依存関係管理にuv（Astral社製の高速Pythonパッケージマネージャー）を使用しています。uvを使うことで、仮想環境の作成や依存関係のインストールが自動化され、環境構築の手間を大幅に削減できます。

### uvのインストール

まず、uvがインストールされていない場合はインストールしてください：

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrewを使用（macOS）
brew install uv

# pipを使用（非推奨、すでにPython環境がある場合）
pip install uv
```

### 依存関係のインストール

```bash
cd scripts/chapter5

# 依存関係のインストール
uv sync

# FAISSも使用する場合（オプション）
uv pip install faiss-cpu
```

これで必要なすべてのパッケージが適切なバージョンでインストールされます。

### サンプルコードの実行

uvを使えば、仮想環境のアクティベートは不要です：

```bash
# 環境のテスト（すべてのライブラリが正しくインストールされているか確認）
uv run python test_imports.py

# OpenAI APIキーを設定
export OPENAI_API_KEY="sk-..."

# スクリプトの実行（uv runを使用）
uv run python 5-1-1-rag-comparison-demo.py
uv run python 5-2-1-search-comparison.py
uv run python 5-3-1-minimal-rag-chroma.py
uv run python 5-4-1-hybrid-search-rrf.py
uv run python 5-5-1-complete-rag-pipeline.py
```

または、.envファイルを作成して実行：

```bash
# .envファイルを作成
cp .env.example .env
# エディタで.envを編集してAPIキーを設定

# 実行
uv run python 5-1-1-rag-comparison-demo.py
```

## ファイル構成

```
chapter5/
├── README.md                         # このファイル
├── pyproject.toml                    # uv用の設定ファイル
├── .env.example                      # 環境変数の設定例
├── test_imports.py                   # 環境セットアップのテスト
├── 5-1-1-rag-comparison-demo.py      # 5.1節: RAGなし/ありの比較デモ
├── 5-2-1-search-comparison.py        # 5.2節: キーワード検索vsベクトル検索の比較
├── 5-3-1-minimal-rag-chroma.py       # 5.3節: 最小RAGデモ（Chroma使用）
├── 5-4-1-hybrid-search-rrf.py        # 5.4節: ハイブリッド検索デモ
├── 5-5-1-complete-rag-pipeline.py    # 5.5節: 統合RAGパイプライン
└── faiss_langchain_demo.py           # 付録: FAISSデモ
```

## セットアップ

### 1. 依存関係のインストール

```bash
cd scripts/chapter5
uv sync

# FAISSを使用する場合（オプション）
uv pip install faiss-cpu  # CPU版
# または
uv pip install faiss-gpu  # GPU版（CUDA環境必須）
```

### 2. OpenAI APIキーの設定

```bash
export OPENAI_API_KEY='your-api-key-here'
```

または `.env` ファイルを作成:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-5-nano  # オプション：使用するモデル
```

## 各スクリプトの実行方法

### 1. RAGなし/ありの比較デモ（5-1-1-rag-comparison-demo.py）

RAGの有無による違いを実感できるデモです。

```bash
# uvを使用（推奨）
uv run python 5-1-1-rag-comparison-demo.py

# または従来の方法
python 5-1-1-rag-comparison-demo.py
```

**学習ポイント:**
- RAGなしでLLMに直接質問した場合の限界
- RAGありで社内文書から回答できることの確認
- 出典の重要性

### 2. キーワード検索vsベクトル検索の比較（5-2-1-search-comparison.py）

キーワード検索（BM25）とベクトル検索（埋め込み）の違いを実際に体験できるデモです。

```bash
# uvを使用（推奨）
uv run python 5-2-1-search-comparison.py

# または従来の方法
python 5-2-1-search-comparison.py
```

**学習ポイント:**
- キーワード検索（BM25）の特徴と限界
- ベクトル検索（埋め込み）による意味的類似性の理解
- エラーコード検索など完全一致が重要な場面の理解
- ハイブリッド検索が必要な理由

### 3. 最小RAGデモ（5-3-1-minimal-rag-chroma.py）

LangChain/LCELを使用した最小構成のRAGシステムです。

```bash
# デフォルトの質問で実行
uv run python 5-3-1-minimal-rag-chroma.py

# カスタム質問で実行
uv run python 5-3-1-minimal-rag-chroma.py "RAGの利点は何ですか？"
```

**学習ポイント:**
- LCELパイプラインの基本構築
- Chromaベクトルストアの使用
- 出典付き回答の生成

### 4. ハイブリッド検索デモ（5-4-1-hybrid-search-rrf.py）

BM25とベクトル検索を組み合わせたハイブリッド検索のデモです。

```bash
uv run python 5-4-1-hybrid-search-rrf.py
```

**学習ポイント:**
- BM25 vs ベクトル検索の比較
- EnsembleRetrieverによるRRF融合
- 重み付けパラメータの影響

### 5. 統合パイプライン（5-5-1-complete-rag-pipeline.py）

完全なRAGシステムの統合実装で、コマンドライン引数で柔軟に設定可能です。

```bash
# 基本的な実行
uv run python 5-5-1-complete-rag-pipeline.py --question "RAGとは何ですか？"

# ハイブリッド検索を使用
uv run python 5-5-1-complete-rag-pipeline.py --question "RRFの効果は？" --hybrid

# FAISSを使用したハイブリッド検索
uv run python 5-5-1-complete-rag-pipeline.py --question "FAISSの特徴" --store faiss --hybrid

# ヘルプの表示
uv run python 5-5-1-complete-rag-pipeline.py --help
```

### 6. FAISSデモ（faiss_langchain_demo.py）- 付録用

LangChainのFAISSラッパを使用したベクトル検索のデモです。

```bash
uv run python faiss_langchain_demo.py
```

**学習ポイント:**
- FAISSインデックスの種類と特性
- save_local/load_localによる永続化
- スコア付き検索とMMR検索

**オプション:**
- `--question`: 質問文を指定
- `--store {chroma,faiss}`: ベクトルストアの種類（デフォルト: chroma）
- `--hybrid`: ハイブリッド検索を使用
- `--data`: カスタムデータファイル/ディレクトリのパス

## サンプルデータ

初回実行時に、`data/ch05/documents.md` にサンプルデータが自動作成されます。独自のデータを使用する場合は、Markdown形式（.md）で準備し、`--data` オプションで指定してください。

## トラブルシューティング

### OpenAI APIキーエラー

```
エラー: OPENAI_API_KEY環境変数が設定されていません
```

**解決方法:**
```bash
export OPENAI_API_KEY='your-api-key'
```

### FAISSインポートエラー

```
ImportError: No module named 'faiss'
```

**解決方法:**
```bash
pip install faiss-cpu
```

### メモリ不足エラー

大きなファイルを処理する場合、chunk_sizeを調整してください:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 小さくする
    chunk_overlap=100
)
```

## パフォーマンスのヒント

1. **開発時**: Chromaで高速プロトタイピング
2. **中規模データ**: FAISSのIndexHNSWを使用
3. **大規模データ**: FAISSのIndexIVFまたはIndexPQを使用
4. **検索精度向上**: ハイブリッド検索（--hybrid）を使用

## 学習の進め方

1. **基礎理解**: `minimal_rag_chroma.py` から始めて、RAGの基本的な流れを理解
2. **ベクトル検索の深掘り**: `faiss_langchain_demo.py` でインデックスの仕組みを学習
3. **検索手法の比較**: `5-5-1-hybrid-search-rrf.py` で各手法の特徴を理解
4. **統合実装**: `run_pipeline.py` で実用的なRAGシステムを体験

## 参考資料

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Chroma Documentation](https://docs.trychroma.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## ライセンス

このコードは教育目的で提供されています。商用利用の際は適切なライセンスを確認してください。
