"""根拠収集と出典提示を行う LangGraph ベースの調査エージェント。"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Annotated, Dict, List, Literal, TypedDict

import operator
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

MAX_STEPS = 8
MIN_SOURCES = 3

SEARCH_MAX_RESULTS = 5
FETCH_TIMEOUT = 8
MAX_DOC_LENGTH = 3200

SYSTEM_PROMPT = """
あなたは調査エージェントです。信頼できる出典を収集し、少なくとも {min_sources} 件の根拠を集めたうえで回答してください。
必要に応じて以下のツールを活用できます。
- web_search: Web検索で候補となるURLを集める
- fetch_page: 指定URLの本文を取得する
出典は URL とともに整理し、重複を避けてください。
"""

OFFLINE_SEARCH_RESULTS: list[dict[str, str]] = [
    {
        "url": "https://www.cross-m.co.jp/column/digital_marketing/dmc20250725",
        "title": "生成AIのリスクと国内外の規制動向",
        "snippet": "日本では生成AIの包括的な規制は未整備だが、各省庁がガイドラインを整備し企業に周知している。",
        "content": (
            "日本では生成AIを直接規制する法律はまだ存在しないが、政府は各省庁を通じてガイドラインを発行している。"
            "企業は個人情報保護法や著作権法など既存の法令を踏まえて運用ルールを整備する必要がある。"
        ),
    },
    {
        "url": "https://shift-ai.co.jp/blog/8657/",
        "title": "世界と日本の生成AI規制まとめ",
        "snippet": "EU の AI Act や米国の大統領令など国際的な規制が進み、日本企業への影響が懸念される。",
        "content": (
            "EU では 2024 年に AI Act が可決され、高リスク AI に対して厳格な義務が課される。"
            "日本企業も国際的な取引や越境データの扱いで影響を受ける可能性がある。"
        ),
    },
    {
        "url": "https://yamamura-law.jp/column/642",
        "title": "生成AIと法的リスク",
        "snippet": "生成AIを巡る日本の法的リスクとして、個人情報保護、著作権、機密情報の取り扱いが挙げられる。",
        "content": (
            "生成AIを業務利用する際は個人情報や機密情報の取り扱いに注意が必要であり、利用規程の整備が求められる。"
        ),
    },
    {
        "url": "https://www.nttdata.com/jp/ja/trends/data-insight/2025/050702/",
        "title": "世界で進むAI規制と企業対応",
        "snippet": "国際的な規制強化を背景に、日本企業もリスク評価とガバナンス整備を急いでいる。",
        "content": (
            "海外の規制動向を踏まえ、日本企業も生成AIの利用範囲や審査体制を整備する必要がある。"
        ),
    },
]

class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    step_count: int
    satisfied: bool
    findings: Dict[str, str]
    sources: Annotated[List[str], operator.add]
    query: str


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"環境変数 {var_name} が設定されていません")
    return value

@tool
def web_search(query: str) -> str:
    """Tavily でクエリ検索を実行し、JSON 文字列で返す"""
    print(f"[Tool:Search] クエリ: {query}")
    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(
            max_results=SEARCH_MAX_RESULTS,
            include_answer=True,
            include_raw_content=False,
        )
        raw = tavily.invoke(query)

        processed: list[dict[str, str | float | None]] = []
        answer = raw.get("answer")
        if answer:
            processed.append(
                {
                    "url": "tavily:answer",
                    "title": "tavily_answer",
                    "snippet": answer,
                    "score": None,
                }
            )

        for item in raw.get("results", []):
            processed.append(
                {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")[:200],
                    "score": item.get("score"),
                }
            )
    else:
        print("[i] TAVILY_API_KEY が未設定のため、Research エージェントはスタブ検索結果を使用します。")
        processed = [
            {
                "url": data["url"],
                "title": data["title"],
                "snippet": data["snippet"],
                "score": None,
            }
            for data in OFFLINE_SEARCH_RESULTS
        ]

    print(f"[Tool:Search] {len(processed)} 件の結果")
    return json.dumps(processed, ensure_ascii=False)

@tool
def fetch_page(url: str) -> str:
    """指定 URL の本文をテキストとして取得する"""
    print(f"[Tool:Fetch] URL: {url}")
    if not os.getenv("TAVILY_API_KEY"):
        print("[i] TAVILY_API_KEY が未設定のため、Fetch はスタブ本文を返します。")
        for data in OFFLINE_SEARCH_RESULTS:
            if data["url"] == url:
                print(f"[Tool:Fetch] オフラインデータを返却 ({len(data['content'])} 文字)")
                return data["content"]
        return "オフラインモードのため、この URL の本文データは含まれていません。"
    try:
        response = requests.get(url, timeout=FETCH_TIMEOUT)
        response.raise_for_status()
    except Exception as err:
        return f"ページの取得に失敗しました: {err}"

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "aside"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)
    if len(text) > MAX_DOC_LENGTH:
        text = text[:MAX_DOC_LENGTH] + "..."

    print(f"[Tool:Fetch] {len(text)} 文字を取得")
    return text


def research_agent_node(state: ResearchState) -> ResearchState:
    print(f"\n[Agent] Step {state['step_count'] + 1}/{MAX_STEPS}")
    print(f"[Agent] 現在の出典数: {len(state['sources'])}")

    new_state = dict(state)
    new_state["messages"] = list(state["messages"])
    new_state["findings"] = dict(state["findings"])
    new_state["sources"] = list(state["sources"])

    if len(state["sources"]) >= MIN_SOURCES:
        print("[Agent] 十分な出典を確保。最終回答を生成")
        llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)

        summary = (
            "以下の情報を基に質問へ回答してください。ポイントは3つ程度にまとめ、各ポイントに出典URLを付与してください。\n"
            f"質問: {state['query']}\n"
        )
        for url, body in state["findings"].items():
            summary += f"\n【出典】{url}\n{body[:500]}\n"

        response = llm.invoke([HumanMessage(content=summary)])
        new_state["messages"].append(response)
        new_state["satisfied"] = True
        new_state["step_count"] = state["step_count"] + 1
        return new_state

    plan_prompt = (
        f"現在 {len(state['sources'])} 件の出典を収集済み。最低 {MIN_SOURCES} 件集める必要があります。"
        "必要に応じて web_search または fetch_page を使用してください。"
    )

    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
    llm = llm.bind_tools([web_search, fetch_page])
    user_instruction = HumanMessage(content=plan_prompt)
    response = llm.invoke([*state["messages"], user_instruction])

    new_state["messages"].extend([user_instruction, response])
    new_state["step_count"] = state["step_count"] + 1
    return new_state


def process_findings_node(state: ResearchState) -> ResearchState:
    if not state["messages"]:
        return state

    new_state = dict(state)
    new_state["messages"] = list(state["messages"])
    new_state["findings"] = dict(state["findings"])
    new_state["sources"] = list(state["sources"])

    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        tool_name = last_message.name or ""
        content = str(last_message.content)

        if tool_name == "web_search":
            try:
                results = json.loads(content)
            except json.JSONDecodeError:
                results = []
            for item in results:
                url = item.get("url")
                snippet = item.get("snippet", "")
                if url and url not in new_state["sources"]:
                    new_state["sources"].append(url)
                    new_state["findings"][url] = snippet
                    print(f"[Process] 新規出典: {url}")
        elif tool_name == "fetch_page":
            prev = state["messages"][-2] if len(state["messages"]) >= 2 else None
            target_url = None
            if isinstance(prev, AIMessage) and prev.tool_calls:
                for call in prev.tool_calls:
                    if call["name"] == "fetch_page":
                        target_url = call.get("args", {}).get("url")
            if target_url:
                new_state["findings"][target_url] = content
                if target_url not in new_state["sources"]:
                    new_state["sources"].append(target_url)
                print(f"[Process] 本文を保存: {target_url}")
        else:
            urls = re.findall(r"https?://\S+", content)
            for url in urls:
                if url not in new_state["sources"]:
                    new_state["sources"].append(url)
                    new_state["findings"].setdefault(url, content[:200])

    return new_state


def should_continue(state: ResearchState) -> Literal["tools", "process", END]:
    if state["step_count"] >= MAX_STEPS:
        print("[Control] ステップ上限に到達")
        return END
    if state["satisfied"]:
        print("[Control] 調査完了")
        return END

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("[Control] ツールノードへ")
        return "tools"
    if isinstance(last_message, ToolMessage):
        print("[Control] 結果処理へ")
        return "process"
    return END


def build_graph() -> StateGraph:
    load_dotenv()
    require_env("OPENAI_API_KEY")

    graph = StateGraph(ResearchState)
    graph.add_node("agent", research_agent_node)
    graph.add_node("tools", ToolNode([web_search, fetch_page]))
    graph.add_node("process", process_findings_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "process": "process",
            END: END,
        },
    )

    graph.add_edge("tools", "process")
    graph.add_edge("process", "agent")

    return graph.compile()


def run(query: str) -> None:
    system_message = SystemMessage(content=SYSTEM_PROMPT.format(min_sources=MIN_SOURCES))
    human_message = HumanMessage(content=query)

    state: ResearchState = {
        "messages": [system_message, human_message],
        "step_count": 0,
        "satisfied": False,
        "findings": {},
        "sources": [],
        "query": query,
    }

    print("=== 調査 Agent：根拠収集と出典提示 ===\n")
    print(f"調査テーマ: {query}")
    print("=" * 60)

    graph = build_graph()
    final_state = graph.invoke(state)

    print("\n" + "=" * 60)
    print("調査結果:\n")

    for message in reversed(final_state["messages"]):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            print(message.content)
            break

    print(f"\n収集した出典数: {len(final_state['sources'])}")
    for idx, url in enumerate(final_state["sources"], start=1):
        print(f"  [{idx}] {url}")


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "2024年の日本における生成 AI 規制の動向"
    run(query)

if __name__ == "__main__":
    main()
