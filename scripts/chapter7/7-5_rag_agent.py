"""社内データ優先で検索し、不足時に Web を補完する RAG × Agent デモ。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import operator
import asyncio
import subprocess
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import ToolNode
import warnings

from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
import requests
from bs4 import BeautifulSoup
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)  # LangChain v1 でモジュール分割された text splitter

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LangGraph の 1 ステップは「LLM 思考 + ツール実行」で2～3カウント進むため、
# 社内+Web+GitHub を行き来する調査でも余裕があるよう 15 ステップ確保しておく。
MAX_STEPS = 15
# LangGraph 自体の recursion_limit はデフォルト 25 と短いため、
# 明示的に大きめのバジェットを確保して agent_node ⇔ tool_node の往復に耐えられるようにする。
GRAPH_RECURSION_LIMIT = 80
FETCH_TIMEOUT = 8
MAX_DOC_LENGTH = 3200

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._migration")

SYSTEM_PROMPT = """
あなたは社内ナレッジを最優先で活用する調査エージェントです。

## ツール選択のガイドライン

1. **まず社内データを確認**: corp_search で社内データを検索してください
   - 検索結果には similarity（類似度、0〜1）が含まれます
   - similarity が 0.75 以上なら信頼できる情報です

2. **情報が不十分な場合の補完**:
   - 社内データの similarity が低い、またはヒットしない場合は追加検索を検討してください
   - 一般的な業界動向・規制情報: web_search を使用
   - 技術的な情報（ライブラリ、フレームワーク、Issue、PR等）: github_search_issues を使用
   - URLの詳細な内容が必要な場合: fetch_page を使用

3. **検索結果の活用**:
   - GitHub検索結果がある場合は、その Issue/PR のタイトルと内容を必ず具体的に引用してください
   - Web検索・GitHub検索で取得した技術的情報は、詳細まで含めて回答に反映してください
   - 「公式ドキュメントを確認してください」のような一般的な回答は避け、取得した具体的な情報を提示してください

4. **最終回答**: 十分な情報が集まったら、各段落の末尾に出典URLまたは出典ファイル名を括弧付きで明記し、『Final Answer:』で回答を開始してください

重要: ツールは必要に応じて複数回、組み合わせて使用できます。
"""

OFFLINE_WEB_DATA = {
    "default": [
        {
            "url": "https://www.weka.io/resources/analyst-report/2024-global-trends-in-ai/",
            "snippet": "Generative AI budgets are projected to reach 34% of total AI spend in 2024 as executives balance regulation and ROI.",
            "content": (
                "A 2024 report highlights growing budgets for generative AI, with organizations balancing regulatory compliance and ROI. "
                "Common themes include governance frameworks, data security, and phased rollouts across business functions."
            ),
        },
        {
            "url": "https://www.deloitte.com/us/en/what-we-do/capabilities/applied-artificial-intelligence/content/state-of-generative-ai-in-enterprise.html",
            "snippet": "Executives pursue innovation while preparing governance playbooks that satisfy emerging global AI regulations.",
            "content": (
                "Enterprises are formalizing AI governance playbooks and risk controls while rolling out GenAI pilots. "
                "Key practices: data anonymization, model evaluation, and cross-functional oversight."
            ),
        },
        {
            "url": "https://www.eversheds-sutherland.com/en/global/insights/global-ai-regulatory-update-february-2024",
            "snippet": "New transparency requirements oblige deployers to disclose training data practices within nine months of enforcement.",
            "content": (
                "Regulatory updates emphasize transparency and accountability, including obligations for providers and deployers around disclosures and incident reporting."
            ),
        },
    ],
    "申請": [
        {
            "url": "https://intra.example.com/ai-usage-guideline",
            "snippet": "社内ポータルで申請後、最大3営業日で承認。顧客データを含むプロンプトは禁止。",
            "content": (
                "社内ガイドラインでは、個人情報・機密情報のプロンプト投入は禁止とされ、匿名化・マスキングが必須。"
                "申請は情報システム部ポータルから行い、リスク評価と監査設定を経て承認される。"
            ),
        },
        {
            "url": "https://intra.example.com/ai-request-flow",
            "snippet": "情報システム部が申請をレビューし、リスク評価とログ監査の設定を行う。",
            "content": (
                "申請フローは、申請→レビュー→承認→利用開始の 4 段階。ログ監査の有効化と責任者承認を経て運用に入る。"
            ),
        },
    ],
}

class RAGAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    step_count: int
    tool_call_count: int
    satisfied: bool
    corp_findings: List[Dict[str, str]]
    web_findings: List[Dict[str, str]]
    confidence: float
    query: str

vector_store: Optional[Chroma] = None
embeddings: Optional[OpenAIEmbeddings] = None
mcp_client: Optional[MultiServerMCPClient] = None
github_tools_enabled: bool = False
active_tool_list: List[Any] = [ ]
github_tool_names: List[str] = []
github_tool_handles: List[Any] = []


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"環境変数 {var_name} が設定されていません")
    return value


def check_gh_auth() -> bool:
    """gh CLI で GitHub 認証済みかを確認"""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_async(coro):
    """asyncio.run の簡易ラッパー（既存ループがある場合にも対応）"""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def initialize_rag() -> None:
    global vector_store, embeddings
    if vector_store is not None and embeddings is not None:
        return

    print("[Init] RAG インデックスを初期化中...")

    require_env("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings()

    sample_docs = [
        {
            "content": "当社の生成AI利用ガイドライン（2024年版）では、顧客データを含むプロンプトを禁止し、機密情報は必ずマスキングすると定めています。",
            "metadata": {"source": "AI利用ガイドライン_v2024.pdf", "page": 12},
        },
        {
            "content": "社内向け ChatGPT 環境は Azure OpenAI Service を利用し、日本リージョンでデータを保管します。外部リークのリスクは低減されています。",
            "metadata": {"source": "システム構成書_202403.pdf", "page": 5},
        },
        {
            "content": "生成AIを業務で利用する際は情報システム部ポータルから申請が必要で、承認には最大3営業日かかります。",
            "metadata": {"source": "申請手順書.pdf", "page": 1},
        },
        {
            "content": "2024年度のAI活用予算は5000万円で、主にライセンス費用とインフラ整備に充当されます。",
            "metadata": {"source": "予算計画書_2024.xlsx", "sheet": "AI投資"},
        },
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )

    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for doc in sample_docs:
        chunks = splitter.split_text(doc["content"])
        texts.extend(chunks)
        metadatas.extend([doc["metadata"]] * len(chunks))

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="corp_knowledge",
    )
    print(f"[Init] {len(texts)} チャンクをインデックス化完了")


@tool
def corp_search(query: str, top_k: int = 3) -> str:
    """社内データを検索し、JSON 文字列を返す"""
    # RAG初期化（初回のみChromaインデックス構築）
    initialize_rag()
    assert vector_store is not None

    # ベクトル検索で類似チャンクを取得（距離スコア付き）
    results = vector_store.similarity_search_with_score(query, k=top_k)

    # 距離→類似度変換して結果を整形
    processed = []
    for doc, score in results:
        similarity = 1 / (1 + float(score))  # 距離→類似度変換
        processed.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "不明"),
                "page": doc.metadata.get("page") or doc.metadata.get("sheet", ""),
                "similarity": similarity,  # 0.0〜1.0の値
            }
        )

    print(f"[Tool:CorpSearch] {len(processed)} 件の結果")
    return json.dumps(processed, ensure_ascii=False)


@tool
def web_search(query: str) -> str:
    """Tavily で補完情報を取得し、JSON 文字列を返す"""
    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(
            max_results=3,
            include_answer=True,
            include_raw_content=False,
        )
        raw = tavily.invoke(query)

        processed = []
        answer = raw.get("answer")
        if answer:
            processed.append({"url": "tavily:answer", "snippet": answer})

        for item in raw.get("results", []):
            processed.append(
                {
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "")[:400],
                }
            )
    else:
        print("[i] TAVILY_API_KEY が未設定のため、Web検索はスタブデータで代用します。")
        key = "申請" if "申請" in query else "default"
        processed = [dict(item) for item in OFFLINE_WEB_DATA.get(key, OFFLINE_WEB_DATA["default"])]

    print(f"[Tool:WebSearch] {len(processed)} 件の結果")
    return json.dumps(processed, ensure_ascii=False)


@tool
def github_search_issues(query: str, state_filter: str = "open", per_page: int = 5) -> str:
    """GitHub の Issues を検索して JSON 形式で返す"""
    if not github_tools_enabled or mcp_client is None:
        return json.dumps(
            {
                "status": "disabled",
                "reason": "GitHub authentication is not available",
            },
            ensure_ascii=False,
        )

    payload = {
        "query": query,
        "state": state_filter,
        "per_page": per_page,
    }

    try:
        raw_response = run_async(mcp_client.call_tool("github", "search_issues", payload))
    except Exception as err:  # noqa: BLE001
        return json.dumps(
            {
                "status": "error",
                "message": str(err),
            },
            ensure_ascii=False,
        )

    if isinstance(raw_response, str):
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            data = {"raw": raw_response}
    else:
        data = raw_response

    items = []
    if isinstance(data, dict):
        possible_items = data.get("items") or data.get("data") or data.get("results")
        if isinstance(possible_items, list):
            items = possible_items
    elif isinstance(data, list):
        items = data

    normalized: List[Dict[str, Any]] = []
    for issue in items[:per_page]:
        if not isinstance(issue, dict):
            continue
        normalized.append(
            {
                "html_url": issue.get("html_url") or issue.get("url", ""),
                "title": issue.get("title", ""),
                "state": issue.get("state", ""),
                "body": (issue.get("body") or "")[:800],
                "number": issue.get("number"),
                "repository_url": issue.get("repository_url"),
            }
        )

    return json.dumps(
        {
            "status": "ok",
            "source": "github_search_issues",
            "items": normalized,
        },
        ensure_ascii=False,
    )


@tool
def fetch_page(url: str) -> str:
    """指定 URL の本文をテキストで取得（オフライン時はスタブ）"""
    print(f"[Tool:Fetch] URL: {url}")
    # オフライン（Tavily キー無し）の場合はスタブを返す
    if not os.getenv("TAVILY_API_KEY"):
        print("[i] TAVILY_API_KEY が未設定のため、Fetch はスタブ本文を返します。")
        for group in OFFLINE_WEB_DATA.values():
            for item in group:
                if item.get("url") == url:
                    body = item.get("content") or item.get("snippet") or ""
                    print(f"[Tool:Fetch] オフライン本文 {len(body)} 文字")
                    return body
        return "オフラインモードのため本文データは見つかりませんでした。"

    # オンライン取得
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
    except Exception as err:
        return f"ページの取得に失敗しました: {err}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for el in soup(["script", "style", "nav", "footer", "aside"]):
        el.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if len(text) > MAX_DOC_LENGTH:
        text = text[:MAX_DOC_LENGTH] + "..."
    print(f"[Tool:Fetch] {len(text)} 文字を取得")
    return text


def agent_node(state: RAGAgentState) -> RAGAgentState:
    """LLMが次のアクションを自律的に決定するノード"""
    print(f"\n[Agent] Step {state['step_count'] + 1}/{MAX_STEPS}")
    print(f"[Agent] 社内データ: {len(state['corp_findings'])} 件 / Web: {len(state['web_findings'])} 件 / 信頼度: {state['confidence']:.2f}")

    # 利用可能なツールを準備
    available_tools = [corp_search, web_search, fetch_page]
    if github_tools_enabled:
        available_tools.append(github_search_issues)

    # LLMにツールをバインド
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.3)
    llm = llm.bind_tools(available_tools)

    # トークン数削減：メッセージ履歴を圧縮
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

    # 現在の状況をLLMに伝える（corp_findingsとweb_findingsの内容）
    if state["corp_findings"] or state["web_findings"]:
        situation_lines = ["## 現在の検索結果サマリー"]

        if state["corp_findings"]:
            situation_lines.append(f"\n### 社内データ（信頼度: {state['confidence']:.2f}）")
            for idx, finding in enumerate(state["corp_findings"][:3], 1):
                source = finding.get("source", "不明")
                content = finding.get("content", "")[:200]
                situation_lines.append(f"{idx}. {source}: {content}")

        if state["web_findings"]:
            situation_lines.append(f"\n### Web データ（{len(state['web_findings'])}件）")
            for idx, finding in enumerate(state["web_findings"][:3], 1):
                url = finding.get("url", "")
                snippet = finding.get("snippet", "")[:150]
                situation_lines.append(f"{idx}. {url}: {snippet}")

        situation_summary = HumanMessage(content="\n".join(situation_lines))
        messages_to_send.append(situation_summary)

    # LLMに次のアクションを決めてもらう
    response = llm.invoke(messages_to_send)

    # 新しい状態を作成
    new_state = dict(state)
    new_state["messages"] = [*state["messages"], response]
    new_state["step_count"] = state["step_count"] + 1
    new_state["corp_findings"] = list(state["corp_findings"])
    new_state["web_findings"] = list(state["web_findings"])

    # ツール呼び出しがない場合は終了フラグを立てる
    if not getattr(response, "tool_calls", None):
        new_state["satisfied"] = True
        print("[Agent] 最終回答を生成")

    return new_state


def process_search_results(state: RAGAgentState) -> RAGAgentState:
    if not state["messages"]:
        return state

    last_message = state["messages"][-1]
    if not isinstance(last_message, ToolMessage):
        return state

    new_state = dict(state)
    new_state["messages"] = list(state["messages"])
    new_state["corp_findings"] = [dict(item) for item in state["corp_findings"]]
    new_state["web_findings"] = [dict(item) for item in state["web_findings"]]
    new_state["tool_call_count"] = state.get("tool_call_count", 0)

    tool_name = last_message.name or ""
    raw_content = last_message.content
    if isinstance(raw_content, str):
        content = raw_content
    else:
        try:
            content = json.dumps(raw_content, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(raw_content)

    if tool_name:
        new_state["tool_call_count"] += 1

    # corp_searchツールの結果を処理：信頼度を計算してstateに反映
    if tool_name == "corp_search":
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            results = []
        new_state["corp_findings"] = results
        if results:
            # 複数チャンクの中で最大類似度を信頼度とする
            best = max(result.get("similarity", 0.0) for result in results)
            new_state["confidence"] = best
            print(f"[Process] 社内データ信頼度を更新: {best:.2f}")
            # ToolMessageを短縮して履歴の肥大化を防ぐ
            try:
                summary = f"社内検索: {len(results)}件ヒット（信頼度: {best:.2f}）"
                new_state["messages"][-1] = ToolMessage(
                    content=summary,
                    name="corp_search",
                    tool_call_id=getattr(last_message, "tool_call_id", None),
                )
            except Exception:
                pass
    elif tool_name == "web_search":
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            results = []
        findings: list[Dict[str, str]] = []
        for item in results:
            snippet = item.get("snippet", "")
            url = item.get("url", "")
            # tavily:answer は擬似URLのため本文取得対象から除外
            if url == "tavily:answer":
                continue
            if snippet or url:
                findings.append({"snippet": snippet, "url": url})
        new_state["web_findings"] = findings
        print(f"[Process] Web 補足情報 {len(findings)} 件")
        # ToolMessageを短縮
        try:
            urls = [f.get("url", "")[:80] for f in findings[:3]]
            summary = f"Web検索: {len(findings)}件取得 - {', '.join(urls)}"
            new_state["messages"][-1] = ToolMessage(
                content=summary,
                name="web_search",
                tool_call_id=getattr(last_message, "tool_call_id", None),
            )
        except Exception:
            pass
    elif tool_name and "github" in tool_name.lower():
        processed = 0
        try:
            result_data = json.loads(content)
        except json.JSONDecodeError:
            result_data = raw_content

        if isinstance(result_data, dict):
            iter_items = [result_data]
        elif isinstance(result_data, list):
            iter_items = result_data
        else:
            iter_items = []

        for item in iter_items[:3]:
            if not isinstance(item, dict):
                continue
            url = item.get("html_url") or item.get("url") or ""
            title = item.get("title") or item.get("name") or ""
            state_label = item.get("state")
            snippet = title or (state_label or "")
            body = item.get("body") or item.get("description") or ""
            content_preview = (body or snippet)[:1000]
            finding = {
                "url": url,
                "snippet": f"{snippet} {('- ' + state_label) if state_label else ''}".strip(),
                "content": content_preview,
            }

            if url and any(existing.get("url") == url for existing in new_state["web_findings"]):
                continue
            new_state["web_findings"].append(finding)
            processed += 1

        if processed:
            print(f"[Process] GitHub結果 {processed} 件を反映")
    elif tool_name == "fetch_page":
        # 直前の AIMessage の tool_calls から対象 URL を推定
        prev = state["messages"][-2] if len(state["messages"]) >= 2 else None
        target_url = None
        if isinstance(prev, AIMessage) and prev.tool_calls:
            for call in prev.tool_calls:
                if call.get("name") == "fetch_page":
                    target_url = call.get("args", {}).get("url")
        body = raw_content if isinstance(raw_content, str) else str(raw_content)
        if target_url:
            updated = False
            for f in new_state["web_findings"]:
                if f.get("url") == target_url:
                    f["content"] = body
                    updated = True
                    break
            if not updated:
                new_state["web_findings"].append({"url": target_url, "content": body})
            print(f"[Process] 本文を保存: {target_url} ({len(body)} 文字)")
            # 大きな ToolMessage は圧縮して履歴を肥大化させない
            try:
                new_state["messages"][-1] = ToolMessage(
                    content=f"本文取得済み: {target_url} ({len(body)} 文字)",
                    name="fetch_page",
                    tool_call_id=getattr(last_message, "tool_call_id", None),
                )
            except Exception:
                pass

    # ToolMessage は本文を圧縮しているため、履歴はそのまま保持（ペア整合性を優先）

    return new_state


def should_continue(state: RAGAgentState) -> Literal["tools", "process", END]:
    # 1. ステップ数上限チェック：暴走防止
    if state["step_count"] >= MAX_STEPS:  # ①
        print("[Control] ステップ上限に達しました")
        return END

    # 2. 満足フラグチェック：agent_nodeで信頼度判定を行い、終了と判断されたか
    if state["satisfied"]:  # ②
        print("[Control] エージェント完了")
        return END

    # 3. 最後のメッセージの種類で次の遷移先を決定
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:  # ③
        # LLMがツール呼び出しを指示 → toolsノードへ
        print("[Control] ツールノードへ")
        return "tools"
    if isinstance(last_message, ToolMessage):  # ④
        # ツール実行結果が返ってきた → processノードへ（結果を解釈）
        print("[Control] 結果処理へ")
        return "process"

    # 4. どれにも当てはまらない場合は終了
    return END  # ⑤


async def build_graph_async() -> StateGraph:
    global mcp_client, github_tools_enabled, active_tool_list, github_tool_names, github_tool_handles

    load_dotenv()
    require_env("OPENAI_API_KEY")

    base_tools: List[Any] = [corp_search, web_search, fetch_page]
    all_tools: List[Any] = list(base_tools)
    github_tools_enabled = False
    github_tool_names = []
    github_tool_handles = []

    if check_gh_auth():
        try:
            print("[Init] GitHub MCP初期化中...")
            mcp_client = MultiServerMCPClient(
                {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "transport": "stdio",
                        "env": {
                            "GITHUB_PERSONAL_ACCESS_TOKEN_USE_GHCLI": "true",
                        },
                    }
                }
            )
            github_tools = await mcp_client.get_tools()
            if github_tools:
                all_tools.extend(github_tools)
                github_tools_enabled = True
                github_tool_names = [tool.name for tool in github_tools]
                github_tool_handles = list(github_tools)
                tool_names = ", ".join(github_tool_names)
                print(f"[Init] GitHub MCPツール {len(github_tools)} 個を有効化: {tool_names}")
                base_tools.append(github_search_issues)
                all_tools.append(github_search_issues)
            else:
                print("[Warning] GitHub MCPツールが取得できませんでした（継続）")
        except Exception as err:
            github_tools_enabled = False
            print(f"[Warning] GitHub MCP初期化失敗（継続）: {err}")
    else:
        print("[Info] GitHub連携: 無効（gh auth login で有効化できます）")

    safe_llm_tools: List[Any] = [corp_search, web_search, fetch_page]
    if github_tools_enabled:
        safe_llm_tools.append(github_search_issues)
    active_tool_list = safe_llm_tools

    graph = StateGraph(RAGAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(all_tools))
    graph.add_node("process", process_search_results)

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


def build_graph() -> StateGraph:
    """同期版ラッパー（LangGraph 互換用）"""
    return asyncio.run(build_graph_async())


def _extract_final_answer(messages: List[BaseMessage]) -> Optional[str]:
    """既存の AIMessage から最終回答を抽出する"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _synthesize_fallback_answer(state: RAGAgentState) -> Optional[str]:
    """十分なツール結果があるのに LLM が Final Answer を返せなかった場合のフェイルセーフ"""
    if not state["corp_findings"] and not state["web_findings"]:
        return None

    context_lines = []
    if state["corp_findings"]:
        context_lines.append("## 社内データ")
        for finding in state["corp_findings"][:4]:
            src = finding.get("source", "不明")
            page = finding.get("page")
            snippet = finding.get("content", "")[:200]
            label = f"{src}" + (f" p.{page}" if page else "")
            context_lines.append(f"- {snippet} ({label})")
    if state["web_findings"]:
        context_lines.append("## Webデータ")
        for finding in state["web_findings"][:4]:
            url = finding.get("url", "不明")
            snippet = finding.get("content") or finding.get("snippet", "")
            context_lines.append(f"- {snippet[:200]} ({url})")

    prompt = (
        "あなたは調査エージェントです。以下の証拠を基に質問に答えてください。\n"
        "必ず『Final Answer:』で始め、各ポイントの末尾に出典（社内資料名やURL）を括弧で示してください。\n"
        "証拠が不足している点は率直に不足と述べて構いません。"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"質問: {state['query']}\n\n証拠:\n" + "\n".join(context_lines),
            ),
        ]
    )
    return response.content if isinstance(response.content, str) else None


def ensure_final_answer(state: RAGAgentState) -> tuple[RAGAgentState, Optional[str]]:
    """最終回答が存在することを保証し、必要ならフェイルセーフで生成する"""
    existing = _extract_final_answer(state["messages"])
    if existing:
        return state, existing

    fallback = _synthesize_fallback_answer(state)
    if not fallback:
        return state, None

    new_state = dict(state)
    new_state["messages"] = [*state["messages"], AIMessage(content=fallback)]
    new_state["satisfied"] = True
    return new_state, fallback


def run(query: str) -> None:
    state: RAGAgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ],
        "step_count": 0,
        "tool_call_count": 0,
        "satisfied": False,
        "corp_findings": [],
        "web_findings": [],
        "confidence": 0.0,
        "query": query,
    }

    print("=== RAG × Agent：社内優先→Web 補完 ===\n")
    print(f"質問: {query}")
    print("=" * 60)

    graph = build_graph()
    final_state = graph.invoke(state, config={"recursion_limit": GRAPH_RECURSION_LIMIT})
    final_state, final_answer = ensure_final_answer(final_state)

    print("\n" + "=" * 60)
    print("最終回答:\n")
    if final_answer:
        print(final_answer)
    else:
        print("Final Answer: 十分な情報が得られませんでした。")

    print(f"\n信頼度: {final_state['confidence']:.2f}")
    print(f"社内データ: {len(final_state['corp_findings'])} 件")
    print(f"Web データ: {len(final_state['web_findings'])} 件")

    # 標準化されたサマリー出力
    steps = final_state["step_count"]
    tool_calls = final_state.get("tool_call_count", 0)

    # sources: 取得したソースの数（社内データ + Web データ）
    sources = len(final_state['corp_findings']) + len(final_state['web_findings'])

    # satisfied: エージェントが満足したかどうか
    satisfied = final_state["satisfied"]

    print(f"\n[Summary] steps={steps} tool_calls={tool_calls} sources={sources} satisfied={satisfied}")


def main() -> None:
    if check_gh_auth():
        print("[✓] GitHub連携: 有効")
    else:
        print("[i] GitHub連携: 無効（gh auth login で有効化可能）")
    print()

    if len(sys.argv) > 1:
        queries = sys.argv[1:]
    else:
        queries = [
            "生成AIを業務で使う際の申請方法を教えてください",
            "2024年の世界的な生成 AI 規制動向を教えてください",
            "Next.js 15 の破壊的変更と移行時の注意点は？",
        ]

    graph = build_graph()
    for query in queries:
        state: RAGAgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query),
            ],
            "step_count": 0,
            "tool_call_count": 0,
            "satisfied": False,
            "corp_findings": [],
            "web_findings": [],
            "confidence": 0.0,
            "query": query,
        }

        print("=== RAG × Agent：社内優先→Web 補完 ===\n")
        print(f"質問: {query}")
        print("=" * 60)

        final_state = graph.invoke(state, config={"recursion_limit": GRAPH_RECURSION_LIMIT})
        final_state, final_answer = ensure_final_answer(final_state)

        print("\n" + "=" * 60)
        print("最終回答:\n")
        if final_answer:
            print(final_answer)
        else:
            print("Final Answer: 十分な情報が得られませんでした。")
        print(f"\n信頼度: {final_state['confidence']:.2f}")
        print(f"社内データ: {len(final_state['corp_findings'])} 件")
        print(f"Web データ: {len(final_state['web_findings'])} 件")

        # 標準化されたサマリー出力
        steps = final_state["step_count"]
        tool_calls = final_state.get("tool_call_count", 0)
        sources = len(final_state['corp_findings']) + len(final_state['web_findings'])
        satisfied = final_state["satisfied"]

        print(f"\n[Summary] steps={steps} tool_calls={tool_calls} sources={sources} satisfied={satisfied}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        sys.stderr.write(str(err) + "\n")
        sys.exit(1)
