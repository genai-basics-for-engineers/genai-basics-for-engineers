# Chapter 2 サンプルコード

第2章「Prompt Engineering」のサンプルコード集です。

## 学習目標

- LLMの基本パラメータ（Temperature、トークン）の理解
- プロンプトパターン（Zero-Shot、Few-Shot、CoT）の実践
- 実務で使えるプロンプト設計スキルの習得

## 新しいファイル構成

### 独立したスクリプト
- `2-1-1.py` - tiktoken によるトークン数計測（LLM API不要）

### LLM API呼び出し用
- `call-llm.py` - 共通のLLM API呼び出しスクリプト
- `prompts/{ID}-prompt.txt` - 各例のプロンプトファイル
- `outputs/{ID}-out.txt` - 実行結果（自動生成）

## 必要な環境

- Python 3.10以上（推奨: 3.11.9）
- OpenAI API キー（[取得方法](https://platform.openai.com/api-keys)）

## 簡単な使い方

### セットアップ
```bash
# uvで依存関係を同期
uv venv
uv pip sync pyproject.toml

# .env を作成して API キーを設定
cp .env.example .env
vi .env  # OPENAI_API_KEY=sk-... を入力

# スクリプトは python-dotenv で .env を自動読み込み
```

### トークン数計測（2-1-1）
```bash
uv run python 2-1-1.py
```

### LLM API呼び出し
```bash
# 基本的な実行
uv run python call-llm.py 2-1-2

# Temperature を変えて実行
uv run python call-llm.py 2-1-2 --temperature 1.5

# 複数回実行して比較
uv run python call-llm.py 2-1-2 --repeat 5 --temperature 0.8

# システムプロンプトを指定
uv run python call-llm.py 2-2-3 --system "あなたは専門家です"
```

### 動作確認
```bash
# 全体テスト
uv run python call-llm.py 2-1-2 --temperature 0.7
```

### 全プロンプトの一括実行
```bash
# 全プロンプトファイルを順番に実行
uv run python run-all.py

# または直接実行
uv run python run-all.py
```

## プロンプトファイル一覧

`prompts/` ディレクトリに以下のIDで保存しています。

| ファイルID | 説明 | 主なテーマ |
|-----------|-----|----------|
| 2-1-2 | Temperature パラメータのデモ | 出力の多様性制御 |
| 2-2-1 | 明確な指示の例 | プロンプトの明確性 |
| 2-2-2 | 肯定的な指示の例 | 否定形 vs 肯定形 |
| 2-2-3 | 技術文書レビュー | System Prompt活用 |
| 2-3-1 | 感情分析 | Few-Shot学習 |
| 2-3-2 | 価格の構造化 | エッジケース対応 |
| 2-4-1 | 数学問題 | Chain-of-Thought |
| 2-4-2 | 在庫管理問題 | ステップバイステップ |

## プロンプトファイルの形式（`prompts/` 配下）

```
---
temperature: 0.7
system: システムプロンプト（オプション）
description: 説明
---
ここにプロンプト本文
```

## 出力ファイルの形式（`outputs/` 配下）

実行結果は `outputs/{ID}-out.txt` に自動保存されます：

```
---
model: gpt-3.5-turbo
temperature: 0.7
max_tokens: 500
executions: 1
has_system_prompt: no
---

LLMの応答がここに記録されます
```

## API キー設定

```bash
# .envファイルを作成
cp .env.example .env

# .envファイルを編集
vim .env  # または好きなエディタで
# OPENAI_API_KEY=sk-... # あなたのAPIキーに置き換え
```

## スクリプト一覧

| ファイル | 内容 | 対応セクション | 実行例 |
|---------|------|--------------|--------|
| `2-0_prompt_examples.py` | プロンプト実例集 | 全体 | `python 2-0_prompt_examples.py --list` |
| `2-1-1_tokenizer_demo.py` | トークン分割の可視化 | 2-1-1 | `python 2-1-1_tokenizer_demo.py` |
| `2-1-1_token_counter.py` | トークン数カウント | 2-1-1 | `python 2-1-1_token_counter.py -t "テキスト"` |
| `2-1-1_token_optimizer.py` | トークン最適化 | 2-1-1 | `python 2-1-1_token_optimizer.py` |
| `2-1-2_temperature_demo.py` | Temperature効果検証 | 2-1-2 | `python 2-1-2_temperature_demo.py --demo` |
| `2-1-3_message_role_demo.py` | メッセージロール比較 | 2-1-3 | `python 2-1-3_message_role_demo.py --system` |
| `2-4_chain_of_thought_demo.py` | CoT効果実証 | 2-4 | `python 2-4_chain_of_thought_demo.py --math` |
| `2-5-2_api_client_generator.py` | APIクライアント生成 | 2-5-2 | `python 2-5-2_api_client_generator.py` |
| `2-5-3_data_analysis_example.py` | データ分析例 | 2-5-3 | `python 2-5-3_data_analysis_example.py --sales` |
| `2-7-1_prompt_evaluator.py` | プロンプト評価システム（シミュレータ版） | 2-7-1 | `python 2-7-1_prompt_evaluator.py` |
| `2-7-2_working_evaluator.py` | プロンプト評価システム（OpenAI API版） | 2-7-2 | `python 2-7-2_working_evaluator.py` |

## 実行例

### Temperature効果の検証

```bash
# デフォルトのデモ（Temperature 0.2, 0.5, 0.7, 0.9で比較）
uv run python 2-1-2_temperature_demo.py --demo

# カスタムプロンプトで実験
uv run python 2-1-2_temperature_demo.py --prompt "明日の天気を詩的に表現してください"
```

### メッセージロールの効果

```bash
# システムロールの効果検証
uv run python 2-1-3_message_role_demo.py --system

# 会話履歴の維持
uv run python 2-1-3_message_role_demo.py --conversation

# API vs UI の違い説明
uv run python 2-1-3_message_role_demo.py --explanation
```

```bash
# 個別に実行
uv run python 2-1-2_temperature_demo.py --demo
uv run python 2-1-3_message_role_demo.py --system
uv run python 2-4_chain_of_thought_demo.py --math
```

## 📖 参考資料

- [OpenAI API ドキュメント](https://platform.openai.com/docs)
- [tiktoken ドキュメント](https://github.com/openai/tiktoken)
- [uv ドキュメント](https://github.com/astral-sh/uv)
- 書籍第2章の該当セクション
