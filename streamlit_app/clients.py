import streamlit as st
import os
from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    CACHE_DISTANCE_THRESHOLD,
    CACHE_EMBED_MODEL,
    CACHE_TTL,
    CLIP_TEXT_MODEL,
    IMAGE_EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    REDIS_URL,
    REDIS_PASSWORD,
    REDIS_USERNAME,
    TEXT_EMBED_MODEL,
)

# Every function here is decorated with @st.cache_resource.
# That decorator creates the object once per Streamlit server process and
# reuses the same instance on every subsequent call — no reconnections,
# no reloading heavyweight model weights on every page interaction.


@st.cache_resource
def get_qdrant() -> QdrantClient:
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=api_key)
    else:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


@st.cache_resource
def get_semantic_cache():
    from redisvl.extensions.llmcache import SemanticCache
    from redisvl.utils.vectorize import HFTextVectorizer

    if REDIS_PASSWORD:
        print(f"Using Redis with authentication")
        return SemanticCache(
            name="pokerag",
            vectorizer=HFTextVectorizer(CACHE_EMBED_MODEL),
            redis_url=REDIS_URL,
            distance_threshold=CACHE_DISTANCE_THRESHOLD,
            ttl=CACHE_TTL,
            overwrite=True,
            connection_kwargs={
                "username": REDIS_USERNAME,
                "password": REDIS_PASSWORD,
            },
        )
    else:
        return SemanticCache(
            name="pokerag",
            vectorizer=HFTextVectorizer(CACHE_EMBED_MODEL),
            redis_url=REDIS_URL,
            distance_threshold=CACHE_DISTANCE_THRESHOLD,
            ttl=CACHE_TTL,
            overwrite=True,
        ) 


@st.cache_resource
def get_llm() -> ChatOllama:
    # LangChain wrapper around the local Ollama server.
    # temperature=0.1 keeps answers factual and consistent rather than creative.
    if os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_API_MODEL"), temperature=0.1)
    else:
        return ChatOllama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL, temperature=0.1)


@st.cache_resource
def get_text_embed_model() -> TextEmbedding:
    # Used for: text-based Pokemon search and semantic cache comparisons.
    # Produces 768-dimensional vectors.
    return TextEmbedding(TEXT_EMBED_MODEL)


@st.cache_resource
def get_clip_model() -> SentenceTransformer:
    # Shared model for both vision and text embeddings (CLIP).
    # laion/CLIP-ViT-H-14 produces 1024-dimensional vectors.
    return SentenceTransformer(IMAGE_EMBED_MODEL)
