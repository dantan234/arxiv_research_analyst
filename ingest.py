import feedparser  # a library that parses arXiv's response format (called "Atom/RSS", a common feed format)
import urllib.request
from shared import embed_text, collection, index


def fetch_arxiv_papers(query="cat:cs.AI", max_results=10):
    """
    Pull recent papers from arXiv matching a search query.
    query examples: 'cat:cs.AI' (category), 'all:RAG' (keyword search)
    """
    base_url = "http://export.arxiv.org/api/query?"
    url = f"{base_url}search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"

    response = urllib.request.urlopen(url)
    feed = feedparser.parse(response.read())

    papers = []
    for entry in feed.entries:
        papers.append({
            "id": entry.id,                      # unique arXiv URL/ID
            "title": entry.title.replace("\n", " ").strip(),
            "abstract": entry.summary.replace("\n", " ").strip(),
            "authors": [a.name for a in entry.authors],
            "published": entry.published,
            "link": entry.link,
        })
    return papers


def prepare_chunks(papers):
    """
    For now, treat each abstract as one chunk.
    Each chunk needs a unique ID and the text to embed.
    """
    chunks = []
    for p in papers:
        chunks.append({
            "id": p["id"],
            "text": f"{p['title']}\n\n{p['abstract']}",
            "metadata": {
                "title": p["title"],
                "link": p["link"],
                "published": p["published"],
                "authors": ", ".join(p["authors"]),
            }
        })
    return chunks


def store_chunks(chunks):
    vectors_to_upsert = []
    for chunk in chunks:
        embedding = embed_text(chunk["text"])
        metadata = dict(chunk["metadata"])
        metadata["text"] = chunk["text"]
        vectors_to_upsert.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": metadata
        })
    index.upsert(vectors=vectors_to_upsert)


if __name__ == "__main__":
    # Pull, chunk, and store new papers
    papers = fetch_arxiv_papers(query="cat:cs.AI", max_results=10)
    chunks = prepare_chunks(papers)
    store_chunks(chunks)

    # Quick sanity check: search for something and see if it finds relevant papers
    query_embedding = embed_text("agents that use tools to solve tasks")
    results = index.query(query_embeddings=[query_embedding], top_k=3)

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(meta["title"])