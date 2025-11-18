#!/usr/bin/env python3
"""
全てのプロンプトファイルを実行するスクリプト
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path("prompts")
OUTPUTS_DIR = Path("outputs")

PROMPTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

def find_prompt_files() -> List[Path]:
    """プロンプトファイルを検索"""
    return sorted(PROMPTS_DIR.glob("*-prompt.txt"))

def run_prompt(file_id: str) -> Tuple[bool, str]:
    """個別のプロンプトを実行"""
    try:
        result = subprocess.run(
            [sys.executable, "call-llm.py", file_id],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output_file = OUTPUTS_DIR / f"{file_id}-out.txt"
            if output_file.exists():
                return True, str(output_file)
            return True, None
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "タイムアウト (30秒)"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 50)
    print("全プロンプト実行スクリプト (Python版)")
    print("=" * 50)
    
    # 環境変数をチェック
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("\n⚠️  OPENAI_API_KEY が設定されていません")
        print("実行前に以下を行ってください:")
        print("  cp .env.example .env && vi .env")
        print("  # OPENAI_API_KEY=... を設定")
        sys.exit(1)
    
    # プロンプトファイルを検索
    prompt_files = find_prompt_files()
    
    if not prompt_files:
        print("❌ プロンプトファイルが見つかりません")
        sys.exit(1)
    
    print(f"\n見つかったプロンプトファイル: {len(prompt_files)}個")
    for path in prompt_files:
        print(f"  - {path}")
    
    print("\n実行を開始します...")
    print("-" * 50)
    
    success_count = 0
    failed_count = 0
    output_files = []
    
    for i, prompt_path in enumerate(prompt_files, 1):
        file_id = prompt_path.name.replace("-prompt.txt", "")
        
        print(f"\n[{i}/{len(prompt_files)}] 実行中: {file_id}")
        
        # プロンプトを実行
        success, result = run_prompt(file_id)
        
        if success:
            success_count += 1
            print(f"  ✅ 成功")
            if result:
                output_files.append(result)
                print(f"  → 出力: {result}")
        else:
            failed_count += 1
            print(f"  ❌ 失敗: {result}")
        
        # API レート制限を考慮
        if i < len(prompt_files):
            print("  ⏳ 1秒待機中...")
            time.sleep(1)
    
    # サマリー表示
    print("\n" + "=" * 50)
    print("実行結果サマリー")
    print("=" * 50)
    print(f"✅ 成功: {success_count}/{len(prompt_files)}")
    print(f"❌ 失敗: {failed_count}/{len(prompt_files)}")
    
    if output_files:
        print(f"\n生成された出力ファイル ({len(output_files)}個):")
        for f in output_files:
            print(f"  - {f}")
    
    print("\n✨ 全プロンプトの実行が完了しました")
    
    # 全て成功した場合は0、失敗がある場合は1を返す
    sys.exit(0 if failed_count == 0 else 1)

if __name__ == "__main__":
    main()
