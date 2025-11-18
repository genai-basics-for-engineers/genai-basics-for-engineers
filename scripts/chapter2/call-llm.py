#!/usr/bin/env python3
"""
共通のLLM API呼び出しスクリプト
プロンプトファイルを読み込んで、LLMの応答を出力ファイルに保存します。

使用例:
    cp .env.example .env && vi .env  # APIキーを設定
    uv run python call-llm.py 2-1-2
    uv run python call-llm.py 2-1-2 --temperature 1.5
    uv run python call-llm.py 2-1-2 --system "あなたは専門家です"
"""

import openai
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path("prompts")
OUTPUTS_DIR = Path("outputs")

PROMPTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def read_prompt_file(file_id: str) -> Dict[str, Any]:
    """
    プロンプトファイルを読み込む
    
    Args:
        file_id: ファイルID (例: "2-1-2")
    
    Returns:
        プロンプト設定の辞書
    """
    prompt_file = PROMPTS_DIR / f"{file_id}-prompt.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"プロンプトファイルが見つかりません: {prompt_file}")
    
    content = prompt_file.read_text(encoding="utf-8")
    
    # メタデータとプロンプトを分離
    lines = content.split("\n")
    metadata = {}
    prompt_start = 0
    
    # メタデータを解析（---で囲まれた部分）
    if lines[0] == "---":
        for i, line in enumerate(lines[1:], 1):
            if line == "---":
                prompt_start = i + 1
                break
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
    
    # プロンプト本文を取得
    prompt = "\n".join(lines[prompt_start:]).strip()
    
    return {
        "prompt": prompt,
        "metadata": metadata
    }

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    model: str = "gpt-5-nano"
) -> str:
    """
    LLM APIを呼び出す
    
    Args:
        prompt: ユーザープロンプト
        system_prompt: システムプロンプト
        temperature: 生成の多様性
        max_tokens: 最大トークン数
        model: 使用するモデル
    
    Returns:
        生成されたテキスト
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your-api-key-here":
        print("⚠️ OPENAI_API_KEY環境変数が設定されていません。")
        print("  1. .env.example をコピーして .env を作成")
        print("  2. .env に OPENAI_API_KEY を設定")
        return "シミュレーション出力：APIキーが設定されていないため、実際のLLM応答は取得できません。"
    
    client = openai.OpenAI(api_key=api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        # モデルに応じてパラメータを設定
        completion_params = {
            "model": model,
            "messages": messages
        }

        # gpt-5-nano系は temperatureとmax_tokensのパラメータを受け付けない
        if "gpt-5" in model:
            # gpt-5-nanoは追加パラメータなし
            pass
        else:
            # 他のモデルは通常のパラメータを使用
            completion_params["temperature"] = temperature
            completion_params["max_tokens"] = max_tokens

        response = client.chat.completions.create(**completion_params)
        return response.choices[0].message.content
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

def save_output(file_id: str, output: str, metadata: Dict[str, Any]):
    """
    出力をファイルに保存
    
    Args:
        file_id: ファイルID
        output: LLMの出力
        metadata: 実行時のメタデータ
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    output_file = OUTPUTS_DIR / f"{file_id}-out.txt"
    
    content = []
    content.append("---")
    for key, value in metadata.items():
        content.append(f"{key}: {value}")
    content.append("---")
    content.append("")
    content.append(output)
    
    output_file.write_text("\n".join(content), encoding="utf-8")
    print(f"✅ 出力を保存しました: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="LLM APIを呼び出してプロンプトを実行"
    )
    parser.add_argument(
        "file_id",
        help="プロンプトファイルのID (例: 2-1-2)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature パラメータ (0.0-2.0)"
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="システムプロンプト"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大トークン数"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="使用するモデル"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="繰り返し実行回数"
    )
    
    args = parser.parse_args()
    
    # デフォルト値を環境変数から取得
    # args.modelは後でfile_metadataと組み合わせて処理するのでここではセットしない
    if args.max_tokens is None:
        args.max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "500"))
    
    print(f"{'='*60}")
    print(f"LLM API 呼び出し: {args.file_id}")
    print(f"{'='*60}")
    
    try:
        # プロンプトファイルを読み込む
        prompt_data = read_prompt_file(args.file_id)
        prompt = prompt_data["prompt"]
        file_metadata = prompt_data["metadata"]
        
        # メタデータからデフォルト値を取得
        temperature = args.temperature or float(file_metadata.get("temperature", 0.7))
        system_prompt = args.system or file_metadata.get("system", None)
        model = args.model or file_metadata.get("model", os.getenv("OPENAI_MODEL", "gpt-5-nano"))

        # 実行回数: コマンドライン引数が1（デフォルト）の場合はメタデータから取得
        if args.repeat == 1 and "executions" in file_metadata:
            args.repeat = int(file_metadata.get("executions", 1))
        
        print(f"\n📄 プロンプト:")
        print("-" * 40)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("-" * 40)
        
        if system_prompt:
            print(f"\n🔧 システムプロンプト:")
            print("-" * 40)
            print(system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt)
            print("-" * 40)
        
        print(f"\n⚙️ パラメータ:")
        print(f"  - Temperature: {temperature}")
        print(f"  - Max Tokens: {args.max_tokens}")
        print(f"  - Model: {model}")
        print(f"  - 実行回数: {args.repeat}")
        
        # LLMを呼び出す
        outputs = []
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"\n🔄 実行 {i+1}/{args.repeat}")
            
            output = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=args.max_tokens,
                model=model
            )
            outputs.append(output)
            
            print(f"\n💬 出力{i+1 if args.repeat > 1 else ''}:")
            print("-" * 40)
            print(output)
            print("-" * 40)
        
        # 出力を保存
        execution_metadata = {
            "model": model,
            "executions": args.repeat,
            "has_system_prompt": "yes" if system_prompt else "no"
        }

        # gpt-5-nano以外はtemperatureとmax_tokensも記録
        if "gpt-5" not in model:
            execution_metadata["temperature"] = temperature
            execution_metadata["max_tokens"] = args.max_tokens
        
        if args.repeat == 1:
            save_output(args.file_id, outputs[0], execution_metadata)
        else:
            # 複数実行の場合はすべての出力を保存
            combined_output = "\n\n=== 実行ごとの出力 ===\n\n".join(
                [f"[実行 {i+1}]\n{out}" for i, out in enumerate(outputs)]
            )
            save_output(args.file_id, combined_output, execution_metadata)
        
    except FileNotFoundError as e:
        print(f"❌ エラー: {e}")
        print(f"\nプロンプトファイル '{PROMPTS_DIR / (args.file_id + '-prompt.txt')}' を作成してください。")
        print("\n例:")
        print("---")
        print("temperature: 0.7")
        print("system: あなたは親切なアシスタントです")
        print("---")
        print("ここにプロンプトを記載")
        sys.exit(1)
    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
