# 第 3 章 環境構築ガイド

本章のサンプルコードを実行するための環境構築手順を説明します。

## 前提条件

- Python 3.13 以上
- コマンドライン操作の基本知識

## セットアップ手順

### 1. uv のインストール

高速な Python パッケージマネージャー「uv」をインストールします。

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

```

インストール後、新しいターミナルセッションを開いて`uv --version`で確認してください。

### 2. プロジェクト環境の準備

```bash
# プロジェクトディレクトリに移動
cd scripts/chapter3

# Python 3.13を指定してプロジェクト環境を初期化
uv python install 3.13
uv python pin 3.13
```

### 3. 依存関係のインストール

```bash
# 必要なライブラリを一括インストール
uv add openai==1.107.3 tiktoken==0.11.0 python-dotenv==1.1.1 requests==2.32.5
```

### 4. 環境変数の設定

```bash
# .envファイルを作成
cp .env.example .env

# APIキーを設定（エディタで開いて編集）
nano .env
```

`.env`ファイルに以下を記入してください：

```
OPENAI_API_KEY=your_api_key_here
```

## 動作確認

環境構築が完了したら、以下のコマンドでサンプルスクリプトを実行できます：

```bash
uv run 3-2-3_env_config.py

```

## トラブルシューティング

### よくある問題と解決方法

- **API キーエラー**: `.env`ファイルに API キーが正しく設定されているか確認
- **パッケージが見つからない**: `uv sync`でパッケージを再同期
- **Python バージョンエラー**: `uv python list`でインストール済みバージョンを確認
