import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# モデル価格
MODEL_PRICES = {
    "gpt-4o-mini": {"in": 0.15, "in_cached": 0.075, "out": 0.6},
    "gpt-5-nano": {"in": 0.05, "in_cached": 0.005, "out": 0.4},
}

class SimpleCostMonitor:
    def __init__(self, budget=5.0, cost_file="./tmp/openai_cost_tracker.json"):
        self.budget = budget
        self.cost_file = Path(cost_file)
        self.total_cost = 0.0
        self.request_count = 0
        self._load()

    def _load(self):
        """累計データを読み込む"""
        if self.cost_file.exists():
            try:
                data = json.loads(self.cost_file.read_text())
                self.total_cost = data.get("total_cost", 0.0)
                self.request_count = data.get("request_count", 0)
            except:
                pass

    def _save(self):
        """累計データを保存"""
        self.cost_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_cost": self.total_cost,
            "request_count": self.request_count
        }
        self.cost_file.write_text(json.dumps(data, indent=2))

    def calculate_cost(self, prompt_tokens, completion_tokens, model="gpt-5-nano", cached_tokens=0):
        """トークン数からコストを計算"""
        p = MODEL_PRICES[model]
        cost = (prompt_tokens * p["in"] + cached_tokens * p["in_cached"] +
                completion_tokens * p["out"]) / 1_000_000
        return cost

    def track(self, prompt_tokens, completion_tokens, model="gpt-5-nano", cached_tokens=0):
        """使用量を追跡してアラート"""
        cost = self.calculate_cost(prompt_tokens, completion_tokens, model, cached_tokens)
        self.total_cost += cost
        self.request_count += 1

        # アラート
        if self.total_cost > self.budget:
            print(f"予算超過: ${self.total_cost:.4f} (予算: ${self.budget})")
        elif self.total_cost > self.budget * 0.8:
            print(f"予算80%到達: ${self.total_cost:.4f}")

        print(f"今回: ${cost:.4f} | 累計: ${self.total_cost:.4f} ({self.request_count}回)")
        self._save()
        return cost

# 使用例
monitor = SimpleCostMonitor(budget=1.0)

def cost_aware_chat(prompt, model="gpt-5-nano"):
    """コスト監視付きチャット"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1000
    )

    monitor.track(
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
        model=model
    )
    return resp.choices[0].message.content

# デモ実行
if __name__ == "__main__":
    result = cost_aware_chat("こんにちは")
    print(f"回答: {result}")
