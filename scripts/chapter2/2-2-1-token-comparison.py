#!/usr/bin/env python3
"""
2-1-1: 日英トークン分割の比較
書籍の解説をそのまま追体験できるよう、ステップごとに計算過程を表示するスクリプト。
"""

from __future__ import annotations

import tiktoken
from typing import List, Sequence, Tuple

Encoding = tiktoken.Encoding

def tokenize(text: str, encoding: Encoding) -> Tuple[List[str], List[int]]:
    token_ids = encoding.encode(text)
    readable_tokens: List[str] = []

    for token_id in token_ids:
        token_bytes = encoding.decode_single_token_bytes(token_id)
        try:
            token_str = token_bytes.decode("utf-8")
            if not token_str:
                raise UnicodeDecodeError("utf-8", token_bytes, 0, len(token_bytes), "empty string")
        except UnicodeDecodeError:
            token_str = "".join(f"\\x{b:02X}" for b in token_bytes)
        readable_tokens.append(token_str)

    return readable_tokens, token_ids

def format_tokens(tokens: Sequence[str]) -> str:
    return "|".join(f"[{token}]" for token in tokens)

def describe_pair(jp_text: str, en_text: str, encoding: Encoding) -> Tuple[int, int]:
    jap_tokens, _ = tokenize(jp_text, encoding)
    eng_tokens, _ = tokenize(en_text, encoding)
    print(f"日本語:「{jp_text}」→{format_tokens(jap_tokens)}({len(jap_tokens)}トークン)")
    print(f"英語:「{en_text}」→{format_tokens(eng_tokens)}({len(eng_tokens)}トークン)")
    if eng_tokens:
        ratio = len(jap_tokens) / len(eng_tokens)
        print(f"比率: {ratio:.2f}倍")
    return len(jap_tokens), len(eng_tokens)

def main() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")

    pairs = [
        ("人工知能", "Artificial Intelligence"),
        ("デバッグする", "Debug code"),
        ("やあ！", "Hello!"),
    ]

    for jp_text, en_text in pairs:
        describe_pair(jp_text, en_text, encoding)
        print()

    world_tokens, _ = tokenize("世界", encoding)
    print(f"日本語:「世界」→{format_tokens(world_tokens)}({len(world_tokens)}トークン)")

if __name__ == "__main__":
    main()
