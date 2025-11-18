#!/usr/bin/env python3
import tiktoken

def main():
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4/3.5用

    japanese_text = "デバッグする"
    english_text = "debug code"

    print(f"日本語: {len(encoding.encode(japanese_text))} トークン")
    print(f"英語: {len(encoding.encode(english_text))} トークン")

if __name__ == "__main__":
    main()
