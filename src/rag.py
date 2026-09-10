import base64
import json
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import ollama

from src.embed import model as embed_model
from src.vectorstore import query
from src.chat import LLM
from src.instructions import get_system_prompt


# ── Utilities ─────────────────────────────────────────────────────────────────

def _content(response: Any) -> str:
    """Normalise LLM wrapper responses to plain text."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("message", {}).get("content", response.get("content", ""))
    message = getattr(response, "message", None)
    if message is not None:
        return getattr(message, "content", "") or str(response)
    return str(response)


def _json_object(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Parse agent JSON, tolerating accidental Markdown code fences."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else fallback
    except json.JSONDecodeError:
        return fallback


def _history_text(chat_history: Optional[Sequence[Dict[str, str]]]) -> str:
    if not chat_history:
        return "(none)"
    lines = []
    for msg in chat_history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _extract_token(chunk: Any) -> str:
    """Extract text token from a streaming chunk (dict or object)."""
    if isinstance(chunk, dict):
        return chunk.get("message", {}).get("content", "")
    msg = getattr(chunk, "message", None)
    if msg is not None:
        return getattr(msg, "content", "")
    return ""


# ── Agent stages ──────────────────────────────────────────────────────────────

def _create_plan(planner: LLM, question: str) -> Dict[str, Any]:
    prompt = (
        "You are the planning agent for a document question-answering system.\n"
        "Return ONLY valid JSON, with no Markdown:\n"
        '{"research_task": "...", "search_queries": ["..."], "answer_goal": "..."}\n\n'
        f"Question: {question}\n"
    )
    fallback = {
        "research_task": question,
        "search_queries": [question],
        "answer_goal": "Answer the user's question from the retrieved documents.",
    }
    plan = _json_object(_content(planner.chat(prompt)), fallback)

    queries = plan.get("search_queries")
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        plan["search_queries"] = [question]
    plan["search_queries"] = (plan["search_queries"][:3] or [question])
    plan.setdefault("research_task", question)
    plan.setdefault("answer_goal", fallback["answer_goal"])
    return plan


def _build_where(source_filter: Optional[List[str]]) -> Optional[dict]:
    if not source_filter:
        return None
    if len(source_filter) == 1:
        return {"source": source_filter[0]}
    return {"source": {"$in": source_filter}}


def _retrieve_evidence(
    researcher: LLM,
    plan: Dict[str, Any],
    n_results: int,
    source_filter: Optional[List[str]] = None,
) -> List[str]:
    """Retrieve candidate chunks then let the researcher agent select the best ones."""
    where = _build_where(source_filter)

    candidates: List[str] = []
    seen: set = set()
    for search_query in plan["search_queries"]:
        embedding = embed_model.encode(search_query)
        for chunk in query(embedding, n_results=n_results, where=where):
            text = str(chunk).strip()
            if text and text not in seen:
                seen.add(text)
                candidates.append(text)

    if not candidates:
        return []

    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
    prompt = (
        "You are the research agent. Select the document chunks that directly "
        "support this task. Return ONLY valid JSON:\n"
        '{"evidence_indices": [0], "research_summary": "brief factual summary"}\n\n'
        f"Task:\n{json.dumps({'research_task': plan['research_task'], 'search_queries': plan['search_queries']})}\n\n"
        f"Candidate chunks:\n{numbered}\n"
    )
    result = _json_object(_content(researcher.chat(prompt)), {})
    selected = result.get("evidence_indices", [])
    if not isinstance(selected, list):
        return candidates
    chosen = [candidates[i] for i in selected if isinstance(i, int) and 0 <= i < len(candidates)]
    return chosen or candidates


def _build_final_prompt(
    question: str,
    plan: Dict[str, Any],
    context: str,
    chat_history: Optional[Sequence[Dict[str, str]]] = None,
) -> str:
    return (
        f"{get_system_prompt()}\n\n"
        "Answer using only the retrieved document evidence below. "
        "If the evidence does not contain the answer, clearly say so. "
        "Do not invent facts.\n\n"
        f"Research goal: {plan['answer_goal']}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Previous conversation:\n{_history_text(chat_history)}\n\n"
        f"Current question: {question}\n\n"
        "Answer:"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def prepare_rag_context(
    question: str,
    chat_history: Optional[Sequence[Dict[str, str]]] = None,
    n_results: int = 3,
    llm_model: str = "gemma2:2b",
    source_filter: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Run the Planner and Researcher stages.
    Returns (plan, context_string, evidence_list).
    Call stream_final_answer() next to get the streamed response.
    """
    planner = LLM(llm_model)
    researcher = LLM(llm_model)
    plan = _create_plan(planner, question)
    evidence = _retrieve_evidence(researcher, plan, n_results, source_filter)
    context = "\n\n---\n\n".join(evidence) if evidence else "(No relevant document chunks were found.)"
    return plan, context, evidence


def stream_final_answer(
    question: str,
    plan: Dict[str, Any],
    context: str,
    chat_history: Optional[Sequence[Dict[str, str]]] = None,
    llm_model: str = "gemma2:2b",
) -> Generator[str, None, None]:
    """Stream the final answer token-by-token given a prepared context."""
    final_prompt = _build_final_prompt(question, plan, context, chat_history)
    stream = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": final_prompt}],
        stream=True,
    )
    for chunk in stream:
        token = _extract_token(chunk)
        if token:
            yield token


def stream_vision_answer(
    image_path: str,
    question: str,
    llm_model: str = "gemma2:2b",
) -> Generator[str, None, None]:
    """Stream an answer to a question about an image (requires a vision model)."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    stream = ollama.chat(
        model=llm_model,
        messages=[{
            "role": "user",
            "content": question,
            "images": [img_b64],
        }],
        stream=True,
    )
    for chunk in stream:
        token = _extract_token(chunk)
        if token:
            yield token


def answer_question(
    question: str,
    chat_history: Optional[Sequence[Dict[str, str]]] = None,
    n_results: int = 3,
    llm_model: str = "gemma2:2b",
    source_filter: Optional[List[str]] = None,
) -> str:
    """Non-streaming version — kept for CLI / backward compatibility."""
    plan, context, _ = prepare_rag_context(
        question, chat_history, n_results, llm_model, source_filter
    )
    final_prompt = _build_final_prompt(question, plan, context, chat_history)
    response = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": final_prompt}],
    )
    return _content(response)


def get_available_models() -> List[str]:
    """Query the local Ollama server for all installed models."""
    try:
        models_info = ollama.list()
        models = (
            models_info.get("models", [])
            if isinstance(models_info, dict)
            else getattr(models_info, "models", [])
        )
        names = []
        for m in models:
            name = m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "")
            if not name and hasattr(m, "name"):
                name = getattr(m, "name", "")
            if name:
                names.append(name)
        return names
    except Exception:
        return []