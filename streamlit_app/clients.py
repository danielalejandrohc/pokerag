import streamlit as st
from fastembed import ImageEmbedding, TextEmbedding
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient

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
    TEXT_EMBED_MODEL,
)

# Every function here is decorated with @st.cache_resource.
# That decorator creates the object once per Streamlit server process and
# reuses the same instance on every subsequent call — no reconnections,
# no reloading heavyweight model weights on every page interaction.


@st.cache_resource
def get_qdrant() -> QdrantClient:
    # Single persistent connection to the Qdrant vector database.
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


@st.cache_resource
def get_semantic_cache():
    from redisvl.extensions.llmcache import SemanticCache
    from redisvl.utils.vectorize import HFTextVectorizer

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
    return ChatOllama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL, temperature=0.1)


@st.cache_resource
def get_text_embed_model() -> TextEmbedding:
    # Used for: text-based Pokemon search and semantic cache comparisons.
    # Produces 384-dimensional vectors.
    return TextEmbedding(TEXT_EMBED_MODEL)


@st.cache_resource
def get_image_embed_model() -> ImageEmbedding:
    # Used for: embedding uploaded images so they can be compared against
    # stored Pokemon artwork vectors. Produces 512-dimensional CLIP vectors.
    return ImageEmbedding(IMAGE_EMBED_MODEL)


@st.cache_resource
def get_clip_text_model() -> TextEmbedding:
    # Used for: "describe what it looks like" search in the image tab.
    # CLIP text and vision encoders share the same vector space, so a text
    # phrase like "yellow electric mouse" can be compared directly against
    # image embeddings without ever looking at the actual images.
    return TextEmbedding(CLIP_TEXT_MODEL)
