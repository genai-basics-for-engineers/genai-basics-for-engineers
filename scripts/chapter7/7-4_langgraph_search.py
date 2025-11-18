"""LangGraph を用いた状態遷移型の検索→要約エージェント。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
from langchain_tavily import TavilySearch

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 5
MIN_SEARCHES = 3

SYSTEM_PROMPT = (
    "あなたはウェブ情報を調査して要約するアシスタントです。\n"
    "重要なルール:\n"
    f"- 必ず最低{MIN_SEARCHES}回は検索ツールを使用してください（異なる検索クエリで）\n"
    "- 異なる観点や複数のソースから情報を収集してください\n"
    "- 最終回答では3つの主要ポイントを箇条書きにまとめ、各ポイントの末尾に出典URLを括弧付きで明記してください\n"
    "- 最終回答は『Final Answer:』で始めてください"
)

OFFLINE_SEARCH_DATA = {
    "answer": (
        "Generative AI adoption accelerated in 2024, driving customer engagement, cost management, and product innovation."
        " Enterprises plan bespoke apps to secure ROI."
    ),
    "results": [
        {
            "url": "https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai",
            "title": "The State of AI: Global survey",
            "content": "Use of generative AI increased from 33% in 2023 to 71% in 2024...",
        },
        {
            "url": "https://blogs.microsoft.com/blog/2024/11/12/idcs-2024-ai-opportunity-study-top-five-ai-trends-to-watch/",
            "title": "IDC's 2024 AI opportunity study",
            "content": "Generative AI powers engagement and cost management; custom line-of-business apps are rising...",
        },
        {
            "url": "https://menlovc.com/2024-the-state-of-generative-ai-in-the-enterprise/",
            "title": "2024: The State of Generative AI in the Enterprise",
            "content": "Generative AI became mission critical as enterprise spend surged...",
        },
    ],
}


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    step_count: int
    search_count: int
    satisfied: bool
    findings: Annotated[List[str], operator.add]

def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"環境変数 {var_name} が設定されていません")
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

def create_search_tool():
    if os.getenv("TAVILY_API_KEY"):
        return TavilySearch(
            max_results=3,
            include_answer=True,
            include_raw_content=False,
        )
    print("[i] TAVILY_API_KEY が未設定のため、LangGraph エージェントはオフライン検索データを使用します。")
    return offline_search

def agent_node(state: AgentState) -> AgentState:
    # LangGraphではstopパラメータを使わないため、gpt-4o-miniが使用可能
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm = llm.bind_tools([create_search_tool()])

    print(f"\n[Agent] Step {state['step_count'] + 1}/{MAX_STEPS}")

    # トークン数削減：最新のHumanMessage以降を保持（会話の一貫性を保つ）
    all_messages = list(state["messages"])
    system_msg = all_messages[0] if all_messages and isinstance(all_messages[0], SystemMessage) else None
    other_messages = all_messages[1:] if system_msg else all_messages

    # 最新のHumanMessage以降を保持（ToolMessage→AIMessageペアを壊さない）
    if len(other_messages) > 0:
        for i in range(len(other_messages) - 1, -1, -1):
            if isinstance(other_messages[i], HumanMessage):
                other_messages = other_messages[i:]
                break

    messages_to_send = ([system_msg] if system_msg else []) + other_messages

    # 検索回数が足りない場合、リマインダーを追加
    if state["search_count"] < MIN_SEARCHES and state.get("satisfied", False):
        reminder = HumanMessage(content=f"注意: 現在{state['search_count']}回しか検索していません。最低{MIN_SEARCHES}回の検索が必要です。異なる検索クエリで追加検索してください。")
        messages_to_send.append(reminder)

    response = llm.invoke(messages_to_send)

    new_state = dict(state)
    # 全履歴は状態に保存（デバッグ用）、LLMには最近のものだけ送信
    new_state["messages"] = [*state["messages"], response]
    new_state["step_count"] = state["step_count"] + 1

    if not getattr(response, "tool_calls", None):
        new_state["satisfied"] = True
        print("[Agent] ツール不要と判断。タスク完了")

    return new_state

def compose_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    new_state = dict(state)

    if isinstance(last_message, ToolMessage):
        new_state["search_count"] = state["search_count"] + 1

        # 検索結果を短縮してトークン数を削減
        parsed_content = last_message.content
        try:
            import json
            # 文字列の場合のみJSON解析を試行
            if isinstance(last_message.content, str):
                try:
                    content = json.loads(last_message.content)
                except json.JSONDecodeError:
                    # JSON解析に失敗した場合はそのまま使用
                    content = last_message.content
            else:
                content = last_message.content

            if isinstance(content, dict) and "results" in content:
                # 各結果のcontentを大幅に短縮してトークン数を削減
                summarized_results = []
                for r in content["results"][:2]:  # 最大2件に制限
                    summarized_results.append({
                        "url": r.get("url", ""),
                        "title": r.get("title", "")[:60],  # タイトルを60文字に短縮
                        "content": r.get("content", "")[:100] + "..."  # 100文字に制限
                    })
                content["results"] = summarized_results
                # answerも大幅に短縮
                if "answer" in content:
                    content["answer"] = content["answer"][:150] + "..."

                # 短縮版のToolMessageで置き換え
                short_message = ToolMessage(
                    content=json.dumps(content, ensure_ascii=False),
                    tool_call_id=last_message.tool_call_id
                )
                new_state["messages"] = [*state["messages"][:-1], short_message]
                parsed_content = content
        except Exception as e:
            print(f"[Compose] 検索結果の短縮に失敗: {e}")

        # 取得したURLを findings に追加（重複除去）
        findings = list(state["findings"])
        try:
            import json
            if isinstance(parsed_content, str):
                try:
                    parsed_content = json.loads(parsed_content)
                except json.JSONDecodeError:
                    pass
            if isinstance(parsed_content, dict) and "results" in parsed_content:
                added = 0
                for r in parsed_content.get("results", []):
                    url = r.get("url") if isinstance(r, dict) else None
                    if url and url not in findings:
                        findings.append(url)
                        added += 1
                if added:
                    new_state["findings"] = findings
                    print(f"[Compose] 新しい情報を追加 (計 {len(findings)} 件)")
        except Exception as e:
            print(f"[Compose] findings への追加に失敗: {e}")

        # 検索クエリを取得（デバッグ用）
        query_info = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                try:
                    query = msg.tool_calls[0].get("args", {}).get("query") or msg.tool_calls[0]["args"]["query"]
                    query_info = f" クエリ: {query}"
                except:
                    pass
                break
        print(f"[Compose] 検索実行 ({new_state['search_count']}回目){query_info}")

    return new_state

def should_continue(state: AgentState) -> Literal["tools", "compose", "agent", END]:
    # 1. ステップ数上限チェック
    if state["step_count"] >= MAX_STEPS:  # ①
        print(f"[Control] ステップ数上限（{MAX_STEPS}）に到達 → 終了")
        return END

    last_message = state["messages"][-1]

    # 2. LLMがツール呼び出しを要求しているか？
    if isinstance(last_message, AIMessage) and last_message.tool_calls:  # ②
        print("[Control] LLM がツール呼び出しを要求 → tools ノードへ遷移")
        return "tools"

    # 3. ツール実行結果を受信したか？
    if isinstance(last_message, ToolMessage):  # ③
        print("[Control] ツール実行結果を受信 → compose ノードへ遷移")
        return "compose"

    # 4. 最低検索回数を満たし、かつ Agent が満足
    if state["search_count"] >= MIN_SEARCHES and state["satisfied"]:  # ④
        print(f"[Control] 最低{MIN_SEARCHES}回の検索完了 & Agent満足 → 終了")
        return END

    # 5. 最低検索回数未達だが Agent が満足している場合
    if state["search_count"] < MIN_SEARCHES and state["satisfied"]:  # ⑤
        print(f"[Control] 検索{state['search_count']}/{MIN_SEARCHES}回、最低回数未達 → 追加検索を要求")
        return "agent"

    return END

def build_graph() -> StateGraph:
    load_dotenv()
    require_env("OPENAI_API_KEY")

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([create_search_tool()]))
    graph.add_node("compose", compose_node)

    graph.set_entry_point("agent")

    # エージェントノードからの条件付き遷移
    graph.add_conditional_edges(
        "agent",  # ① 起点ノード
        should_continue,  # ② 判定関数
        {
            "tools": "tools",
            "compose": "compose",
            "agent": "agent",
            END: END,
        },
    )

    # 固定の遷移パス
    graph.add_edge("tools", "compose")  # ③ ツール実行 → 結果処理
    graph.add_edge("compose", "agent")  # ④ 結果処理 → 次の思考

    return graph.compile()

def run(query: str) -> None:
    app = build_graph()

    state: AgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)],
        "step_count": 0,
        "search_count": 0,
        "satisfied": False,
        "findings": [],
    }

    print("=== LangGraph による検索・要約 Agent ===\n")
    print(f"質問: {query}")
    print("=" * 60)

    final_state = app.invoke(state)

    print("\n" + "=" * 60)
    print("最終回答:\n")

    for message in reversed(final_state["messages"]):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            print(message.content)
            break

    info_count = len(final_state["findings"])
    step_count = final_state["step_count"]
    tool_calls = final_state["search_count"]  # 実行した検索回数と一致
    sources = info_count  # 収集したURL数をそのまま採用
    satisfied = final_state["satisfied"]

    print(f"\n収集した情報数: {info_count}")
    print(f"実行ステップ数: {step_count}")
    print(f"\n[Summary] steps={step_count} tool_calls={tool_calls} sources={sources} satisfied={satisfied}")

def main() -> None:
    user_query = sys.argv[1] if len(sys.argv) > 1 else "2025年の生成 AI 業界の主要な動向を3つ教えてください"
    run(user_query)

if __name__ == "__main__":
    main()
