import hashlib
import json
import time

# シンプルなインメモリキャッシュ（Redisの代替）
cache_store = {}

def mock_llm_api_call(prompt, model_params):
    """模擬LLM API呼び出し"""
    # 実際のAPI呼び出しをシミュレート
    print(f"API呼び出し実行中... プロンプト: {prompt[:50]}...")
    time.sleep(0.5)  # API遅延をシミュレート

    return {
        "response": f"'{prompt}'に対する回答です。",
        "tokens_used": len(prompt.split()) * 2,
        "model": model_params.get("model", "gpt-5-nano")
    }

def get_cached_response(prompt, model_params=None):
    if model_params is None:
        model_params = {"model": "gpt-5-nano"}

    # キャッシュキー生成
    cache_key = hashlib.md5(
        f"{prompt}_{str(model_params)}".encode()
    ).hexdigest()

    # キャッシュチェック
    if cache_key in cache_store:
        cache_data = cache_store[cache_key]
        # TTLチェック（24時間 = 86400秒）
        if time.time() - cache_data["timestamp"] < 86400:
            print("キャッシュから取得")
            return cache_data["response"]
        else:
            # 期限切れキャッシュを削除
            del cache_store[cache_key]

    # キャッシュミス時はAPI呼び出し
    print("キャッシュミス - API呼び出し")
    response = mock_llm_api_call(prompt, model_params)

    # キャッシュに保存
    cache_store[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }

    return response

# 使用例
if __name__ == "__main__":
    # 1回目の呼び出し（キャッシュミス）
    result1 = get_cached_response("Pythonについて教えて")
    print(f"結果1: {result1['response']}\n")

    # 2回目の呼び出し（キャッシュヒット）
    result2 = get_cached_response("Pythonについて教えて")
    print(f"結果2: {result2['response']}\n")

    # パラメータが違う場合（キャッシュミス）
    result3 = get_cached_response("Pythonについて教えて", {"model": "gpt-4"})
    print(f"結果3: {result3['response']}\n")

    # キャッシュの状況確認
    print(f"キャッシュエントリ数: {len(cache_store)}")
