import os
import chromadb
import time
from google import genai
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="arxiv_papers")

pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# index = pinecone_client.Index("arxiv-papers")


def call_with_retry(func, max_retries=5, delay=10):
    """Retry a function call if it fails, with a pause between attempts."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # out of retries, let the error surface
            print(f"Attempt {attempt + 1} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)


def embed_text(text):
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values
