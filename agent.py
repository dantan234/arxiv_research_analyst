from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from shared import embed_text, collection, gemini_client
import time


class AgentState(TypedDict):
    question: str
    retrieved_chunks: List[dict]
    answer: str


def call_with_retry(func, max_retries=5, delay=5):
    """Retry a function call if it fails, with a pause between attempts."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # out of retries, let the error surface
            print(f"Attempt {attempt + 1} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)


def retrieve(state: AgentState) -> dict:
    question = state["question"]
    query_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "metadata": meta})

    return {"retrieved_chunks": chunks}


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

    response = call_with_retry(
        lambda: gemini_client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=user_prompt,
            config={"system_instruction": SYSTEM_PROMPT}
        )
    )
    return {"answer": response.text}



graph_builder = StateGraph(AgentState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("synthesize", synthesize)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "synthesize")
graph_builder.add_edge("synthesize", END)

agent = graph_builder.compile()

if __name__ == "__main__":
        
    # for m in gemini_client.models.list():
    #     print(m.name)

    result = agent.invoke({"question": "What are recent approaches to reducing hallucination in RAG systems?"})
    print(result["answer"])