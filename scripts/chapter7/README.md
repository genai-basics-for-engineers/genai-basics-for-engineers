# Chapter 7 サンプルコード

第7章「AI Agent 開発編」で紹介するコードと実行ログのまとめです。章内で引用するコードはすべてこのディレクトリから取得し、実際に実行した結果を `*-out.txt` に保存しています。

## 🎯 このフォルダでできること

- LangChain の ReAct エージェントを動かし、検索→要約までを追体験
- LangGraph で状態遷移を明示化したエージェント実装をステップごとに確認
- 根拠付き回答や RAG 連携エージェントを実行し、取得したログをそのまま本文に反映
- 付録コード（ガードレールや評価用スニペット）をまとめて参照

## 📦 依存パッケージとセットアップ

`pyproject.toml` / `uv.lock` に依存ライブラリを定義しています。uv を使えば次の 2 ステップで環境構築と API キー設定が完了します。

```bash
cd scripts/chapter7
uv sync --python 3.13  # 仮想環境 (.venv) を生成して依存をインストール

cp .env.example .env
vi .env  # OPENAI_API_KEY / TAVILY_API_KEY を入力
```

スクリプト側で `load_dotenv()` を呼んでいるため、`.env` は自動で読み込まれます。Python は 3.13 系を前提としています。

## 🚀 実行方法の例

### 7-3 ReAct エージェント（検索→要約）
```bash
uv run python 7-3_react_search.py | tee outputs/7-3_react_search-out.txt
```

### 7-4 LangGraph エージェント（状態管理付き）
```bash
uv run python 7-4_langgraph_search.py | tee outputs/7-4_langgraph_search-out.txt
```

### 7-5 調査エージェント（出典提示）
```bash
uv run python 7-5_research_agent.py | tee outputs/7-5_research_agent-out.txt
```

### 7-6 RAG × Agent（社内優先→Web 補完）
```bash
uv run python 7-6_rag_agent.py | tee outputs/7-6_rag_agent-out.txt
```

## 🧪 一括実行

```bash
uv run python run_all.py
```

すべてのデモを順番に実行し、最新のログを `outputs/` 配下に保存します。各スクリプト自身でも `scripts/chapter7/outputs/` を自動生成するため、個別実行・一括実行のどちらでも同じ場所にまとめられます。

## 📁 ファイル一覧

| ファイル | 説明 |
|----------|------|
| `7-3_react_search.py` | LangChain ReAct を使った検索→要約デモ |
| `7-4_langgraph_search.py` | LangGraph による状態遷移付きエージェント |
| `7-5_research_agent.py` | 根拠収集と出典提示を行う調査エージェント |
| `7-6_rag_agent.py` | 社内データ優先の RAG × Agent 実装 |
| `run_all.py` | 主要デモを順番に実行し出力をまとめるスクリプト |
| `pyproject.toml` / `uv.lock` | uv 用の依存管理ファイル |
| `.env.example` | 必須 API キーのサンプル |
| `outputs/` | 実行結果ログ（各スクリプト・`run_all.py` が自動生成） |

## 📝 補足

- 章本文では `scripts/chapter7` からコードを引用し、`outputs/` に保存した実行結果を掲載します。
- ネットワーク環境や API 制限により、一部のログは内容が変わることがあります。章のスクリーンショットを更新するときは、最新のログを取得して差し替えてください。
- GitHub Issues/PR を補完情報として使いたい場合は、事前に `gh auth login` を実行しておくと `7-6_rag_agent.py` が MCP サーバーを自動的に有効化します。
- 出力先ディレクトリは各スクリプトと同じ場所（`scripts/chapter7/outputs/`）に固定しており、`cd scripts/chapter7` 済みでも階層が二重に掘られることはありません。
