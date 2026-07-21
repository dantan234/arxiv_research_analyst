from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from shared import embed_text, index, gemini_client, generate_with_fallback
import time


class AgentState(TypedDict):
    question: str
    is_comparison: bool
    sub_queries: List[str]
    retrieved_chunks: List[dict]
    answer: str


def retrieve(state: AgentState) -> dict:
    all_chunks = []
    seen_ids = set()
    for sub_q in state["sub_queries"]:
        query_embedding = embed_text(sub_q)
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        for match in results["matches"]:
            if match["id"] not in seen_ids:
                all_chunks.append({"text": match["metadata"]["text"], "metadata": match["metadata"]})
                seen_ids.add(match["id"])
    return {"retrieved_chunks": all_chunks}


def classify_question(state: AgentState) -> dict:
    question = state["question"]
    prompt = f"""Is this question asking to compare two or more specific things (e.g. papers, methods, approaches)?
    If yes, break it into separate sub-questions, one per thing being compared.
    If no, just return the original question as the only sub-question.

    Question: {question}

    Respond in this exact format:
    IS_COMPARISON: yes or no
    SUB_QUERIES: sub-question 1 | sub-question 2 | ...
    """
    response = generate_with_fallback(prompt)
    text = response.text.strip()

    is_comparison = "yes" in text.split("IS_COMPARISON:")[1].split("\n")[0].lower()
    sub_queries_raw = text.split("SUB_QUERIES:")[1].strip()
    sub_queries = [q.strip() for q in sub_queries_raw.split("|")]

    return {"is_comparison": is_comparison, "sub_queries": sub_queries}


def synthesize(state: AgentState) -> dict:
    SYSTEM_PROMPT = """You are a research assistant answering questions about arXiv papers.

    Prioritize and cite the retrieved papers provided as context when they're relevant.
    You may also use your general knowledge to add helpful background or context
    (e.g., explaining established concepts), but clearly distinguish this from claims
    grounded in the retrieved papers - for example, say "According to [Paper Title]..."
    for grounded claims, versus "More generally, ..." for outside knowledge.

    If the retrieved context doesn't contain relevant information for the question,
    say so honestly rather than guessing."""

    question = state["question"]
    chunks = state["retrieved_chunks"]

    # Build a context block from the retrieved papers
    context_parts = []
    for c in chunks:
        title = c["metadata"]["title"]
        link = c["metadata"]["link"]
        context_parts.append(f"Title: {title}\nLink: {link}\nText: {c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Context (retrieved papers):
    {context}

    Question: {question}"""

    response = generate_with_fallback(user_prompt, system_instruction=SYSTEM_PROMPT)
    return {"answer": response.text}



graph_builder = StateGraph(AgentState)

graph_builder.add_node("classify_question", classify_question)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("synthesize", synthesize)

graph_builder.add_edge(START, "classify_question")
graph_builder.add_edge("classify_question", "retrieve")
graph_builder.add_edge("retrieve", "synthesize")
graph_builder.add_edge("synthesize", END)

agent = graph_builder.compile()

if __name__ == "__main__":
    # Test a normal question
    result = agent.invoke({"question": "What is retrieval-augmented generation?"})
    print("--- Normal question ---")
    print(result["answer"])
    print()

    # Test a comparison question - use two paper titles actually in your index
    result2 = agent.invoke({"question": "Compare the approach in CRAFT: Clustering Rubrics to Diagnose Weak LLM Capabilities with the approach in Understanding Reasoning from Pretraining to Post-Training"})
    print("--- Comparison question ---")
    print("Sub-queries generated:", result2.get("sub_queries"))
    print(result2["answer"])