#!/usr/bin/env python
"""
環境セットアップのテストスクリプト

必要なライブラリがすべてインポートできるか確認します。
"""

import sys
print(f"Python version: {sys.version}")

try:
    import langchain
    print(f"✓ langchain: {langchain.__version__}")
except ImportError as e:
    print(f"✗ langchain: {e}")

try:
    import langchain_openai
    print("✓ langchain_openai")
except ImportError as e:
    print(f"✗ langchain_openai: {e}")

try:
    import langchain_community
    print("✓ langchain_community")
except ImportError as e:
    print(f"✗ langchain_community: {e}")

try:
    import chromadb
    print(f"✓ chromadb: {chromadb.__version__}")
except ImportError as e:
    print(f"✗ chromadb: {e}")

try:
    import openai
    print(f"✓ openai: {openai.__version__}")
except ImportError as e:
    print(f"✗ openai: {e}")

try:
    import rank_bm25
    print("✓ rank_bm25")
except ImportError as e:
    print(f"✗ rank_bm25: {e}")

try:
    import faiss
    print("✓ faiss-cpu")
except ImportError as e:
    print(f"✗ faiss-cpu: {e}")

print("\n環境チェック完了！")