import os
import chromadb
import time
import random
from google import genai
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="arxiv_papers")

pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pinecone_client.Index("arxiv-papers")


def call_with_retry(func, max_retries=8, base_delay=10, max_delay=300):
    """
    Retry a function call with exponential backoff + jitter.
    Designed for unattended jobs where waiting longer is fine,
    as long as it eventually succeeds.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # out of retries, let it fail

            # exponential backoff: 10s, 20s, 40s, 80s, 160s, 300s (capped), ...
            delay = min(base_delay * (2 ** attempt), max_delay)
            # add jitter: randomize +/- 20% so retries don't all sync up
            jitter = delay * random.uniform(-0.2, 0.2)
            wait_time = delay + jitter

            print(f"Attempt {attempt + 1}/{max_retries} failed ({e}), retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)


def embed_text(text):
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values
