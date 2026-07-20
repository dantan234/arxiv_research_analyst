import os
import requests
from dotenv import load_dotenv
from shared import gemini_client, collection, call_with_retry
from ingest import fetch_arxiv_papers, prepare_chunks, store_chunks

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]


def summarize_paper(paper: dict) -> str:
    """Ask Gemini to write a short, plain-language summary of one paper."""
    prompt = f"""Summarize this paper's abstract in 2-3 plain-language sentences,
    avoiding jargon where possible, for a reader who wants a quick sense of what it's about
    and why it might matter.

    Title: {paper['title']}
    Abstract: {paper['abstract']}

    Summary:"""

    response = call_with_retry(
        lambda: gemini_client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
        )
    )
    return response.text.strip()


def build_digest_message(papers: list, summaries: list) -> str:
    """Format papers + summaries into one Discord message."""
    lines = [f"**📄 ArXiv AI Digest — {len(papers)} new papers today**\n"]
    for paper, summary in zip(papers, summaries):
        lines.append(f"**{paper['title']}**\n{summary}\n{paper['link']}\n")
    return "\n".join(lines)


def send_to_discord(message: str):
    """POST the digest message to the Discord webhook."""
    # Discord has a 2000 character limit per message - split if needed
    max_len = 1900
    for i in range(0, len(message), max_len):
        chunk = message[i:i + max_len]
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk})
        response.raise_for_status()  # raises an error if the request failed


if __name__ == "__main__":
    papers = fetch_arxiv_papers(query="cat:cs.AI", max_results=5)

    # Grow the shared Chroma corpus
    chunks = prepare_chunks(papers)
    store_chunks(chunks)

    # Summarize each paper and send the digest
    summaries = [summarize_paper(p) for p in papers]
    message = build_digest_message(papers, summaries)
    send_to_discord(message)

    print("Digest sent!")