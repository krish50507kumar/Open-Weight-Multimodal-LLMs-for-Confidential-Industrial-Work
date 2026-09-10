# import ollama
# from src.embed import model as embed_model
# from src.vectorstore import query
# from src.chat import LLM
#
# def answer_question(question, chat_history=None, n_results=3, llm_model="gemma2:2b"):
#     planner = LLM(llm_model)
#     researcher = LLM(llm_model)
#     plan = planner.chat(question)
#     question_embedding = embed_model.encode(question)
#     retrieved_chunks = query(question_embedding, n_results=n_results)
#     context = "\n\n".join(retrieved_chunks)
#
#     history_text = ""
#     if chat_history:
#         recent = chat_history[-6:]
#         for msg in recent:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             history_text += f"{role}: {msg['content']}\n"
#
#     prompt = f"""Answer using only the context below. If the context doesn't contain the answer, say so.
#
# Context:
# {context}
#
# Previous conversation:
# {history_text if history_text else "(none)"}
#
# Current question: {question}
#
# Answer:"""
#
#     response = ollama.chat(
#         model=llm_model,
#         messages=[{'role': 'user', 'content': prompt}]
#     )
#     return response['message']['content']
#
# def get_available_models():
#     try:
#         models_info = ollama.list()
#         return [m['model'] for m in models_info['models']]
#     except Exception:
#         return ["gemma3:4b"]

import json
from typing import Any, Dict, List, Optional, Sequence

import ollama

from src.embed import model as embed_model
from src.vectorstore import query
from src.chat import LLM


def _content(response: Any) -> str:
    """Normalise common LLM wrapper responses to text."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("message", {}).get("content", response.get("content", ""))
    # ollama SDK returns a ChatResponse-like object, not a dict
    message = getattr(response, "message", None)
    if message is not None:
        return getattr(message, "content", "") or str(response)
    return str(response)


def _json_object(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Parse an agent's JSON, tolerating accidental Markdown code fences."""
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
    for message in chat_history[-6:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)


def _create_plan(planner: LLM, question: str) -> Dict[str, Any]:
    prompt = f"""You are the planning agent for a document question-answering system.
Return ONLY valid JSON, with no Markdown:
{{"research_task": "...", "search_queries": ["..."], "answer_goal": "..."}}

Question: {question}
"""
    fallback = {
        "research_task": question,
        "search_queries": [question],
        "answer_goal": "Answer the user's question from the retrieved documents.",
    }
    plan = _json_object(_content(planner.chat(prompt)), fallback)

    # Keep the hand-off safe even when the model returns incomplete JSON.
    queries = plan.get("search_queries")
    if not isinstance(queries, list) or not all(isinstance(item, str) for item in queries):
        plan["search_queries"] = [question]
    plan["search_queries"] = plan["search_queries"][:3] or [question]
    plan.setdefault("research_task", question)
    plan.setdefault("answer_goal", fallback["answer_goal"])
    return plan


def _retrieve_evidence(researcher: LLM, plan: Dict[str, Any], n_results: int) -> List[str]:
    """Retrieve candidates from ChromaDB, then let the researcher select evidence."""
    candidates: List[str] = []
    seen = set()
    for search_query in plan["search_queries"]:
        embedding = embed_model.encode(search_query)
        for chunk in query(embedding, n_results=n_results):
            text = str(chunk).strip()
            if text and text not in seen:
                seen.add(text)
                candidates.append(text)

    if not candidates:
        return []

    numbered_candidates = "\n\n".join(
        f"[{index}] {chunk}" for index, chunk in enumerate(candidates)
    )
    handoff = {
        "research_task": plan["research_task"],
        "search_queries": plan["search_queries"],
    }
    prompt = f"""You are the research agent. Select the document chunks that directly
support this structured task. Return ONLY valid JSON:
{{"evidence_indices": [0], "research_summary": "brief factual summary"}}

Task:
{json.dumps(handoff)}

Candidate chunks:
{numbered_candidates}
"""
    result = _json_object(_content(researcher.chat(prompt)), {})
    selected_indices = result.get("evidence_indices", [])
    if not isinstance(selected_indices, list):
        return candidates

    selected = [
        candidates[index]
        for index in selected_indices
        if isinstance(index, int) and 0 <= index < len(candidates)
    ]
    return selected or candidates


def answer_question(
    question: str,
    chat_history: Optional[Sequence[Dict[str, str]]] = None,
    n_results: int = 3,
    llm_model: str = "gemma2:2b",
) -> str:
    """Answer a question with explicit planner, researcher, and answer stages."""
    planner = LLM(llm_model)
    researcher = LLM(llm_model)

    plan = _create_plan(planner, question)
    evidence = _retrieve_evidence(researcher, plan, n_results)
    context = "\n\n---\n\n".join(evidence) or "(No relevant document chunks were found.)"

    final_prompt = f"""Answer using only the retrieved document evidence below.
If the evidence does not contain the answer, clearly say so. Do not invent facts.

Research goal: {plan['answer_goal']}

Retrieved evidence:
{context}

Previous conversation:
{_history_text(chat_history)}

Current question: {question}

Answer:"""
    response = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": final_prompt}],
    )
    return _content(response)


def get_available_models() -> List[str]:
    try:
        models_info = ollama.list()
        models = models_info.get("models", []) if isinstance(models_info, dict) else models_info.models
        return [model["model"] if isinstance(model, dict) else model.model for model in models]
    except (AttributeError, KeyError, TypeError, OSError):
        return ["gemma3:4b"]