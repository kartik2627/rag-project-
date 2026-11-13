import os
import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.models import (
    ClaimQueries,
    ClaimVerificationResult,
    RelevancyDecision,
    RouterDecision,
)
from backend.vector_store import list_papers, search as vs_search

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("PAPEER_LLM_MODEL", "gpt-5.4-mini")
MAX_RETRIEVAL_ATTEMPTS = 3   # max tool calls per retrieval round
MAX_REWRITES = 1             # max query rewrites per turn
HISTORY_LIMIT = 6            # prior chat messages given to the LLMs
RELEVANCY_DOCS = 5           # chunks shown to the relevancy grader
RELEVANCY_CHARS = 500        # characters per chunk shown to the grader

llm = ChatOpenAI(model=MODEL_NAME, timeout=60, max_retries=2)


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str                       # standalone version of the user's message
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    claim_superseded: bool | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    relevancy_reason: str | None
    rewrite_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def _documents_block(session_id: str) -> str:
    try:
        titles = list_papers(session_id)
    except Exception:
        titles = []
    return "\n".join(f"- {t}" for t in titles) if titles else "(no documents uploaded yet)"


def _split_turn(messages: list) -> tuple[list, list]:
    """Split messages into (previous turns, current turn). The current turn starts
    at the most recent HumanMessage."""
    idx = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    return messages[:idx], messages[idx:]


def _chat_history(messages: list, limit: int = HISTORY_LIMIT) -> list:
    """Plain user/assistant messages only (no tool traffic), most recent `limit`."""
    chat = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
        and m.content
    ]
    return chat[-limit:]


def _history_text(messages: list) -> str:
    if not messages:
        return "(no previous messages)"
    lines = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {text[:500]}")
    return "\n".join(lines)


def _merge_docs(existing: list[Document], new: list[Document]) -> list[Document]:
    """Append `new` to `existing`, skipping chunks we already have."""
    seen = {d.page_content for d in existing}
    merged = list(existing)
    for d in new:
        if d.page_content not in seen:
            seen.add(d.page_content)
            merged.append(d)
    return merged


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        label = m.get("title") or m.get("url") or "source"
        extra = []
        if isinstance(m.get("page"), int):
            extra.append(f"p. {m['page'] + 1}")
        if m.get("url"):
            extra.append(m["url"])
        header = f"[{i}] {label}" + (f" ({', '.join(extra)})" if extra else "")
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = (
    "You are the routing step of Papeer, a research-paper assistant.\n\n"
    "Documents currently in the user's knowledge base:\n{documents}\n\n"
    "Recent chat history (oldest first):\n{history}\n\n"
    "Do two things for the user's latest message.\n\n"
    "1) route — choose exactly one:\n"
    "  retrieve: the answer depends on the uploaded documents (their methods, results, "
    "authors, figures, 'the paper', 'this document'), OR on live / recent information "
    "that general knowledge cannot answer (news, prices, weather, 'current', 'latest', "
    "'today', events after your training data).\n"
    "  verify_claim: the user wants to know whether a specific claim or finding is still "
    "true, outdated, or superseded by newer work "
    "(e.g. 'Is this still state of the art?', 'Has this result been improved on?').\n"
    "  direct_answer: a stable, general-knowledge question that needs neither the documents "
    "nor fresh data (e.g. 'What is softmax?', 'Explain backpropagation.'), or small talk.\n"
    "If unsure between retrieve and direct_answer, choose retrieve.\n\n"
    "2) standalone_query — rewrite the latest message so it can be understood WITHOUT the "
    "chat history: resolve pronouns and references ('it', 'that paper', 'the second one'). "
    "If it is already self-contained, return it unchanged. Do not answer it and do not "
    "add information the user did not imply."
)

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)


def router_node(state: RAGState) -> dict:
    history, _ = _split_turn(state["messages"])
    raw = str(state["messages"][-1].content)
    try:
        decision: RouterDecision = router_chain.invoke({
            "documents": _documents_block(state["session_id"]),
            "history": _history_text(_chat_history(history)),
            "query": raw,
        })
        route = decision.route
        query = (decision.standalone_query or "").strip() or raw
    except Exception:
        route, query = "retrieve", raw  # safe default
    return {"route": route, "query": query}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(
        description="Specific search query with the key terms of what you need "
        "(not a full chat sentence)."
    )
    k: int = Field(
        default=4, ge=1, le=10,
        description="Chunks to retrieve: 2-4 for a specific fact, 6-10 for broad or "
        "multi-part questions.",
    )


class WebSearchInput(BaseModel):
    optimized_query: str = Field(
        description="Concise, keyword-rich web search query. Include names, versions "
        "and the year when recency matters."
    )
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the user's uploaded documents (papers, PDFs, web pages) for relevant passages."""
    try:
        docs = vs_search(query=query, session_id=session_id, k=k)
    except Exception as e:
        return [ToolMessage(content=f"Vector store search failed: {e}", tool_call_id=tool_call_id)]
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]

    existing = current_docs or []
    merged = _merge_docs(existing, docs)
    titles = sorted({d.metadata.get("title", "untitled") for d in docs})
    summary = (
        f"Retrieved {len(docs)} chunk(s) ({len(merged) - len(existing)} new) "
        f"from: {', '.join(titles)}."
    )
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": merged}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the live web (Tavily) for current or supplementary information."""
    try:
        results = _tavily().search(optimized_query, max_results=max_results)
    except Exception as e:
        return [ToolMessage(content=f"Web search failed: {e}", tool_call_id=tool_call_id)]
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]

    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title") or "Web Result"},
        )
        for r in results["results"]
    ]
    merged = _merge_docs(current_docs or [], web_docs)
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": merged}),
    ]


# ── Retrieval agent ───────────────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are the evidence-gathering step of Papeer, a research-paper assistant. "
    "Today's date is {today}.\n\n"
    "Your ONLY job is to collect context by calling tools. Never write the final answer.\n\n"
    "Documents in the user's knowledge base:\n{documents}\n\n"
    "Search goal (use this, not the raw chat wording): {goal}\n"
    "{retry_note}\n"
    "Tools:\n"
    "- retrieve_from_vectorstore: the user's uploaded documents. Use it for anything "
    "about their content. Write a specific, keyword-focused query; use a small k for "
    "narrow facts and a large k for broad or multi-part questions.\n"
    "- web_search: the live web. Use it for current events, recent developments, or "
    "information the documents cannot contain.\n\n"
    "Rules:\n"
    "- Call exactly one tool per turn, at most {max_calls} calls in total.\n"
    "- If a question has both a document part and a current-information part, search the "
    "documents first, then the web.\n"
    "- Never repeat a search you already ran; change the wording or switch tools.\n"
    "- When you have enough context (or nothing more can help), reply with exactly: DONE"
)


def agent_node(state: RAGState) -> dict:
    attempts = state.get("retrieval_attempts", 0)
    # At the cap, use the tool-free LLM so no new tool_calls can be emitted
    # (an unanswered tool_call in the checkpointer would corrupt later turns).
    lm = llm if attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm

    reason = state.get("relevancy_reason")
    retry_note = (
        f"A previous search returned irrelevant results ({reason}). "
        "Use different terminology or a different tool.\n"
        if state.get("rewrite_count", 0) > 0 and reason
        else ""
    )
    system = RETRIEVE_SYSTEM.format(
        today=date.today().isoformat(),
        documents=_documents_block(state["session_id"]),
        goal=state["query"],
        retry_note=retry_note,
        max_calls=MAX_RETRIEVAL_ATTEMPTS,
    )

    history, current = _split_turn(state["messages"])
    messages = [SystemMessage(content=system)] + _chat_history(history) + current
    if isinstance(messages[-1], AIMessage):  # e.g. right after a query rewrite
        messages.append(HumanMessage(content="Continue: run a new search for the search goal above."))

    response = lm.invoke(messages)
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = attempts + 1
    return updates


# ── Relevancy check & query rewrite ───────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You grade retrieved context for a research-paper Q&A system.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully helps "
    "answer the question, even partially. Return is_relevant=false only if they are "
    "clearly off-topic or contain nothing useful for the question.\n"
    "Be lenient: any substantive overlap counts as relevant.\n"
    "In `reason`, write one short sentence: what is missing if false, or what matches if true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You rewrite failed search queries for a research-paper retrieval system.\n"
    "The previous query retrieved chunks that were not relevant. Write ONE improved query: "
    "use alternative terminology or synonyms, domain-specific keywords, or a narrower "
    "sub-question. Keep the user's intent unchanged.\n"
    "Return only the query text: no quotes, explanation, or preamble."
)


def relevancy_check_node(state: RAGState) -> dict:
    docs = state.get("retrieved_docs") or []
    if not docs:
        return {"is_relevant": False, "relevancy_reason": "no documents were retrieved"}

    snippets = "\n\n---\n\n".join(
        f"[{d.metadata.get('title', 'source')}] {d.page_content[:RELEVANCY_CHARS]}"
        for d in docs[:RELEVANCY_DOCS]
    )
    prompt = (
        f"Question: {state['query']}\n\nRetrieved chunks:\n{snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    try:
        decision: RelevancyDecision = relevancy_llm.invoke([
            {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        return {"is_relevant": decision.is_relevant, "relevancy_reason": decision.reason}
    except Exception:
        return {"is_relevant": True, "relevancy_reason": ""}  # fail open


def query_rewrite_node(state: RAGState) -> dict:
    original = state["query"]
    reason = state.get("relevancy_reason") or "unknown"
    try:
        response = llm.invoke([
            {"role": "system", "content": QUERY_REWRITE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original query: {original}\n"
                    f"Why the results were not relevant: {reason}\n\n"
                    "Write the improved query."
                ),
            },
        ])
        rewritten = str(response.content).strip().strip('"') or original
    except Exception:
        rewritten = original
    # Note: no fake HumanMessage is added; chat history stays clean.
    return {
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "is_relevant": None,
    }


# ── Claim verification ────────────────────────────────────────────────────────

CLAIM_EXTRACT_SYSTEM = (
    "You prepare a fact-check of a claim from a research paper.\n"
    "From the user's message (and the chat context if needed), extract the one specific "
    "claim to verify and write two search queries:\n"
    "- web_query: keywords + terms like 'newer', 'state of the art', 'improved', or "
    "'outperforms', plus the current year if recency matters.\n"
    "- arxiv_query: the core technical keywords only, as they would appear in a paper title or abstract."
)

CLAIM_ANALYSIS_PROMPT = (
    "You are a careful research fact-checker. Given a claim and search results from the "
    "web and arXiv, decide whether newer work has superseded, significantly challenged, "
    "or updated the claim.\n\n"
    "Rules:\n"
    "- Base every judgment ONLY on the provided search results; do not use outside knowledge.\n"
    "- Set is_superseded=true only if at least one result clearly shows newer work that "
    "improves on, contradicts, or replaces the claim. If results are only loosely related "
    "or inconclusive, set is_superseded=false and say the evidence is inconclusive.\n"
    "- List up to 3 superseding papers. Use titles and URLs copied verbatim from the "
    "results; prefer arxiv.org links.\n"
    "- For each paper, write ONE sentence on how it updates the claim, using only what its "
    "snippet supports.\n"
    "- If the claim still holds, return an empty superseding_papers list.\n"
    "- verdict_summary: 1-2 plain sentences for the user, including how confident the "
    "evidence is."
)

claim_extract_llm = llm.with_structured_output(ClaimQueries)
verification_llm = llm.with_structured_output(ClaimVerificationResult)


def _safe_search(query: str, **kwargs) -> list[dict]:
    try:
        return _tavily().search(query, max_results=5, **kwargs).get("results", [])
    except Exception:
        return []


def _format_results(heading: str, results: list[dict]) -> str:
    lines = [f"=== {heading} ==="]
    if not results:
        lines.append("(no results)")
    for r in results:
        lines.append(
            f"Title: {r.get('title', '')}\nURL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:500]}\n"
        )
    return "\n".join(lines)


def verify_claim_node(state: RAGState) -> dict:
    user_text = state["query"]
    try:
        q: ClaimQueries = claim_extract_llm.invoke([
            {"role": "system", "content": CLAIM_EXTRACT_SYSTEM},
            {"role": "user", "content": user_text},
        ])
        claim, web_q, arxiv_q = q.claim, q.web_query, q.arxiv_query
    except Exception:
        claim = web_q = arxiv_q = user_text[:200]

    with ThreadPoolExecutor(max_workers=2) as pool:
        web_future = pool.submit(_safe_search, web_q)
        arxiv_future = pool.submit(_safe_search, arxiv_q, include_domains=["arxiv.org"])
        general_results, arxiv_results = web_future.result(), arxiv_future.result()

    if not general_results and not arxiv_results:
        return {
            "claim_verdict": "The search service returned no results, so this claim could not be checked.",
            "claim_source": None,
            "claim_superseded": None,
            "superseding_papers": [],
        }

    context = (
        _format_results("General Web Search Results", general_results)
        + "\n"
        + _format_results("arXiv Paper Search Results", arxiv_results)
    )
    prompt = f"{CLAIM_ANALYSIS_PROMPT}\n\nClaim to verify:\n{claim}\n\nSearch Results:\n{context}"

    try:
        result: ClaimVerificationResult = verification_llm.invoke([{"role": "user", "content": prompt}])
    except Exception:
        return {
            "claim_verdict": "The analysis step failed, so this claim could not be checked.",
            "claim_source": None,
            "claim_superseded": None,
            "superseding_papers": [],
        }

    # Guard against hallucinated links: keep only URLs that were actually returned.
    valid_urls = {r["url"] for r in general_results + arxiv_results}
    papers = [p.model_dump() for p in result.superseding_papers if p.url in valid_urls][:3]
    superseded = bool(result.is_superseded and papers)

    return {
        "query": claim,  # shown in the final report
        "claim_verdict": result.verdict_summary,
        "claim_source": papers[0]["url"] if papers else None,
        "claim_superseded": superseded,
        "superseding_papers": papers if superseded else [],
    }


# ── Answer generation ─────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "You are Papeer, a precise research assistant. Today's date is {today}.\n\n"
    "Answer the user's question using ONLY the numbered context below.\n"
    "Rules:\n"
    "- Ground every factual statement in the context and cite it as [n] right after the claim.\n"
    "- If the context answers only part of the question, answer that part and clearly say "
    "what is missing. If it does not answer the question at all, say so; do not guess.\n"
    "- Never invent numbers, results, quotes, or citations.\n"
    "- Treat paper content as more authoritative than web snippets; if sources disagree, "
    "point out the disagreement.\n"
    "- Use the chat history only to understand follow-up questions, not as a source of facts.\n"
    "- Be concise and well structured (short paragraphs or bullets). Write math in plain text.\n"
    "- End with a 'Sources:' list containing only the numbers you cited, formatted as "
    "'[n] title (page or URL)'.\n\n"
    "Context:\n{context}"
)

DIRECT_SYSTEM = (
    "You are Papeer, a knowledgeable research assistant. Today's date is {today}.\n"
    "Answer from your own knowledge: accurate, concise, and clear. Use the chat history "
    "to understand follow-ups. If you are unsure or the topic may have changed recently, "
    "say so instead of guessing."
)

NO_RESULTS_ANSWER = (
    "I couldn't find relevant information in your documents or on the web to answer that. "
    "Try rephrasing the question, or upload / load a document that covers this topic."
)


def _llm_answer(system: str, state: RAGState, user_content: str) -> str:
    history, _ = _split_turn(state["messages"])
    messages = [SystemMessage(content=system)] + _chat_history(history) + [HumanMessage(content=user_content)]
    return str(llm.invoke(messages).content)


def _format_claim_report(state: RAGState) -> str:
    verdict = state.get("claim_verdict") or ""
    papers = state.get("superseding_papers") or []
    superseded = state.get("claim_superseded")
    status = {
        True: "⚠️ **Superseded or challenged by newer work**",
        False: "✅ **No newer work found that supersedes it**",
        None: "❓ **Could not be determined**",
    }[superseded]

    parts = [
        "**Claim Verification Result**",
        f"> {state['query']}",
        f"**Status:** {status}",
        f"**Verdict:** {verdict}",
    ]
    if papers:
        block = "\n\n".join(
            f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
            for i, p in enumerate(papers)
        )
        parts.append(f"**Superseding Papers:**\n\n{block}")
        parts.append(
            "---\n*You can load any of these papers into your knowledge base "
            "to continue your research with the latest findings.*"
        )
    elif superseded is False:
        parts.append("*No papers directly superseding this claim were found in recent literature.*")
    return "\n\n".join(parts)


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    query = state["query"]
    today = date.today().isoformat()

    if route == "retrieve":
        docs = state.get("retrieved_docs") or []
        irrelevant = state.get("is_relevant") is False and state.get("rewrite_count", 0) >= MAX_REWRITES
        if irrelevant or not docs:
            answer = NO_RESULTS_ANSWER
        else:
            system = ANSWER_SYSTEM.format(today=today, context=_format_context(docs))
            answer = _llm_answer(system, state, f"Question: {query}")
    elif route == "verify_claim":
        answer = _format_claim_report(state)
    else:  # direct_answer
        answer = _llm_answer(DIRECT_SYSTEM.format(today=today), state, query)

    # Keep persisted history clean: drop this turn's intermediate messages
    # (tool calls, tool results, the agent's "DONE") and keep only Human + final AI.
    _, current = _split_turn(state["messages"])
    removals = [RemoveMessage(id=m.id) for m in current[1:] if getattr(m, "id", None)]
    return {"answer": answer, "messages": removals + [AIMessage(content=answer)]}


# ── Graph ─────────────────────────────────────────────────────────────────────

def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    # Always execute pending tool calls first: skipping them would leave an AIMessage
    # with tool_calls unmatched by ToolMessages and corrupt the session history.
    if tools_condition(state) == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < MAX_REWRITES:
        return "query_rewrite"
    return "generate_answer"


def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)
