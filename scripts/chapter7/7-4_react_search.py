"""LangChain の ReAct パターンで Web 検索→要約を行うサンプル。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.errors import GraphRecursionError

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LangGraph では LLM ステップと Tool ステップの両方がカウントされるため、
# ReAct の 3～4 回のツール利用を許容するようデフォルト 12 ステップを確保しておく。
DEFAULT_RECURSION_LIMIT = 12

OFFLINE_SEARCH_DATA = {
    "answer": (
        "2024 年の生成 AI 業界では導入率が 33% から 71% へ拡大し、顧客エンゲージメントやコスト管理、"
        "製品開発での活用が急増しています。ROI を得た企業はカスタム アプリケーション投資を加速しました。"
    ),
    "results": [
        {
            "url": "https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai",
            "title": "The State of AI: Global survey",
            "content": "Use of generative AI increased from 33% in 2023 to 71% in 2024. Use of AI by business function ...",
        },
        {
            "url": "https://blogs.microsoft.com/blog/2024/11/12/idcs-2024-ai-opportunity-study-top-five-ai-trends-to-watch/",
            "title": "IDC's 2024 AI opportunity study",
            "content": "Generative AI drives customer engagement, topline growth, and cost management; custom deployments are rising...",
        },
        {
            "url": "https://menlovc.com/2024-the-state-of-generative-ai-in-the-enterprise/",
            "title": "2024: The State of Generative AI in the Enterprise",
            "content": "Generative AI became mission critical in 2024 as enterprise spending surged and bespoke solutions emerged...",
        },
    ],
}


def require_env(var_name: str) -> str:
    """環境変数を取得し、存在しなければエラー終了する"""
    value = os.getenv(var_name)
    if not value:
        sys.stderr.write(f"環境変数 {var_name} が設定されていません\n")
        sys.exit(1)
    return value


@tool("offline_search")
def offline_search(query: str) -> dict:
    """サンプルデータを返すオフライン検索ツール"""
    return {
        "query": query,
        "follow_up_questions": None,
        "answer": OFFLINE_SEARCH_DATA["answer"],
        "images": [],
        "results": OFFLINE_SEARCH_DATA["results"],
    }


def describe_tools(tools: Iterable[BaseTool]) -> str:
    """ツール一覧を system prompt 用に整形する"""
    lines = []
    for tool_obj in tools:
        name = getattr(tool_obj, "name", tool_obj.__class__.__name__)
        description = getattr(tool_obj, "description", "") or getattr(tool_obj, "__doc__", "") or "説明なし"
        description = " ".join(description.strip().split())
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def build_system_prompt(tools: Iterable[BaseTool]) -> str:
    """create_agent 用の system prompt を生成する"""
    tool_section = describe_tools(tools)
    return textwrap.dedent(
        f"""
        あなたは調査と要約を担当するアシスタントです。以下のルールを厳守して質問に答えてください。

        1. 必ず最初のアクションでツールを使って検索し、最新情報を取得すること。
        2. ツールを使わずに結論を出さないこと。
        3. 回答は必ず検索結果に基づき、出典URLを各ポイントの末尾に半角括弧で記載すること。
        4. 最終回答は「Final Answer: 」で始め、日本語で簡潔にまとめること。

       利用可能なツール:
        {tool_section}

        Thought/Action/Observation 形式で行動計画を示し、十分な根拠が集まったら Final Answer を出力してください。
        """
    ).strip()


def build_agent(model: str = "gpt-4o-mini") -> Runnable:
    """ReAct エージェントを構築する"""
    load_dotenv()
    openai_api_key = require_env("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    print(f"\n=== 検証設定 ===")
    print(f"モデル: {model}")
    print("=" * 60 + "\n")

    llm = ChatOpenAI(
        model=model,
        temperature=0.3,  # 調査・制御系のため低めに設定
        api_key=openai_api_key,
    )

    if tavily_api_key:
        search_tool = TavilySearch(
            max_results=3,
            include_answer=True,
        )
    else:
        print("[i] TAVILY_API_KEY が未設定のため、ReAct エージェントはオフライン検索データを使用します。")
        search_tool = offline_search

    tools = [search_tool]
    # LangChain v1 では create_agent + system prompt が公式推奨構成のため、
    # ReAct 要件（初手で検索し Final Answer を整形）を system prompt に集約する。
    system_prompt = build_system_prompt(tools)

    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
    )


def _content_to_text(content: object) -> str:
    """LangChain Message.content を文字列へ正規化する"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    texts.append(str(block["text"]))
                elif block.get("type") == "text" and "content" in block:
                    texts.append(str(block["content"]))
        return "".join(texts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def _extract_result_count(payload: object) -> int:
    """ToolMessage payload から results 配列の長さを推定する（LangGraph v1 互換）。"""

    def _unwrap(data: object) -> Optional[dict]:
        if isinstance(data, dict):
            if "results" in data:
                return data
            # LangGraph v1 の block では `json` や `content` キー配下に実データが入る
            for key in ("json", "content"):
                if key in data:
                    nested = _unwrap(data[key])
                    if nested:
                        return nested
        elif isinstance(data, list):
            for item in data:
                nested = _unwrap(item)
                if nested:
                    return nested
        elif isinstance(data, str):
            try:
                loaded = json.loads(data)
            except json.JSONDecodeError:
                return None
            return _unwrap(loaded)
        return None

    data = _unwrap(payload)
    if isinstance(data, dict):
        results = data.get("results", [])
        if isinstance(results, list):
            return len(results)
    return 0


def summarize_sources(tool_messages: Iterable[ToolMessage]) -> int:
    """ToolMessage 群から取得 URL 数を合計する。"""
    return sum(_extract_result_count(message.content) for message in tool_messages)


def compute_summary_stats(result: dict, final_answer: str) -> tuple[int, int, int]:
    """LangGraph v1 でも steps / tool_calls / sources を推定できるようにする。"""
    intermediate_steps = result.get("intermediate_steps") or []
    if intermediate_steps:
        steps = len(intermediate_steps)
        tool_calls = sum(
            1
            for step in intermediate_steps
            if isinstance(step, (list, tuple)) and len(step) >= 2 and step[0] is not None
        )
        sources = sum(
            _extract_result_count(step[1])
            for step in intermediate_steps
            if isinstance(step, (list, tuple)) and len(step) >= 2 and step[1] is not None
        )
        return steps, tool_calls, sources

    # LangGraph create_agent は intermediate_steps を返さないため、messages から推定する。
    messages = result.get("messages", [])
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    tool_calls = len(tool_messages)
    sources = summarize_sources(tool_messages)

    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    action_messages = [m for m in ai_messages if getattr(m, "tool_calls", None)]
    if action_messages:
        steps = len(action_messages) + tool_calls
    else:
        steps = tool_calls * 2 if tool_calls else 0
        if final_answer:
            steps += 1
    return steps, tool_calls, sources


def main(
    query: Optional[str] = None,
    model: str = "gpt-4o-mini",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> None:
    agent = build_agent(model=model)
    question = query or "2025年の生成 AI 業界の主要な動向を3つ教えてください"

    print("=== ReAct Agent による検索・要約 ===\n")
    print(f"質問: {question}\n")
    print("=" * 60)

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": recursion_limit},
        )
    except GraphRecursionError as exc:
        print("\n" + "=" * 60)
        print("[Error] LangGraph の再帰上限に達したため途中で停止しました。")
        print(
            "ヒント: `--max-steps` で値を大きくするか、質問を具体化して再実行してください "
            f"(現在の設定: {recursion_limit})."
        )
        print(f"詳細: {exc}")
        return
    messages = result.get("messages", [])
    final_message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    final_answer = _content_to_text(final_message.content).strip() if final_message else ""
    # LangGraph v1 create_agent は result["output"] に値を入れないため messages 側の値を優先し、
    # 従来 AgentExecutor 互換で output にのみ格納されるケースはフォールバックで吸収する。
    if not final_answer:
        output_text = result.get("output", "")
        final_answer = _content_to_text(output_text).strip() if output_text else ""
    if final_answer and not final_answer.startswith("Final Answer:"):
        final_answer = f"Final Answer: {final_answer}"

    print("\n" + "=" * 60)
    print("最終回答:\n")
    print(final_answer)

    # 標準化されたサマリー出力
    steps, tool_calls, sources = compute_summary_stats(result, final_answer)

    # satisfied: 何らかの最終回答が生成されたかどうか
    satisfied = bool(final_answer)

    print(
        f"\n[Summary] steps={steps} tool_calls={tool_calls} sources={sources} "
        f"satisfied={satisfied} limit={recursion_limit}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Agent による検索・要約")
    parser.add_argument("query", nargs="?", help="質問（省略時はデフォルト質問を使用）")
    parser.add_argument("--model", default="gpt-4o-mini", help="使用するLLMモデル（デフォルト: gpt-4o-mini）")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_RECURSION_LIMIT,
        help="LangGraph の recursion_limit を指定。Tool/LLM ステップ合計の上限 (デフォルト: 12)",
    )
    args = parser.parse_args()
    main(query=args.query, model=args.model, recursion_limit=args.max_steps)
