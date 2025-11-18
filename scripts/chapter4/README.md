# 第 4 章 環境構築ガイド

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
cd scripts/chapter4

# Python 3.13を指定してプロジェクト環境を初期化
uv python install 3.13
uv python pin 3.13
```

### 3. 依存関係のインストール

```bash
# 必要なライブラリを一括インストール
uv add langchain==1.0.3 langchain-openai==1.0.1 python-dotenv==1.2.1 langchain-community==0.4.1 langgraph==1.0.2
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
uv run 4-2-1_env_config.py

```

## JupyterLab の起動方法

JupyterLab を使用してサンプルコードを実行する場合は、以下の手順で起動してください：

### 1. JupyterLab のインストール

```bash
# JupyterLabをプロジェクトに追加
uv add jupyterlab==4.4.10
```

### 2. JupyterLab の起動

```bash
# JupyterLabサーバーを起動
uv run jupyter lab
```

実行すると、ブラウザが自動的に開き、JupyterLab のインターフェースが表示されます。
もしブラウザが開かない場合は、ターミナルに表示される URL をコピーしてブラウザで開いてください。

### 3. ノートブックファイルの実行

1. JupyterLab のファイルブラウザで、実行したい`.ipynb`ファイルをクリック
2. ノートブックが開いたら、各セルを順番に実行
3. セルを実行するには：
   - `Shift + Enter`：現在のセルを実行し、次のセルに移動
   - `Ctrl + Enter`：現在のセルを実行し、同じセルに留まる

### 4. 終了方法

- JupyterLab を終了するには、ターミナルで`Ctrl + C`を 2 回押す
- またはブラウザの JupyterLab タブで「Quit」ボタンをクリック

## トラブルシューティング

### よくある問題と解決方法

- **API キーエラー**: `.env`ファイルに API キーが正しく設定されているか確認
- **パッケージが見つからない**: `uv sync`でパッケージを再同期
- **Python バージョンエラー**: `uv python list`でインストール済みバージョンを確認
