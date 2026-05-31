from io import BytesIO
import logging

import streamlit as st
from langchain_core.tools import tool
from PIL import Image
from qdrant_client.models import FieldCondition, Filter, MatchValue

from clients import (
    get_clip_model,
    get_qdrant,
    get_text_embed_model,
)
from config import (
    COLLECTION_NAME,
    IMAGE_PAYLOAD,
    IMAGE_SCORE_THRESHOLD,
    MAX_RESULTS,
    TEXT_PAYLOAD,
    SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ── Readiness ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def check_ready() -> bool:
    try:
        existing = {c.name for c in get_qdrant().get_collections().collections}
        return COLLECTION_NAME in existing
    except Exception:
        return False


# ── App search (main tabs) ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def search_text(query: str):
    logger.info(f"Searching for Pokémon matching text query: {query}")
    vec = next(iter(get_text_embed_model().embed([query]))).tolist()
    return get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        using="text",
        limit=MAX_RESULTS,
        with_payload=TEXT_PAYLOAD,
        score_threshold=SCORE_THRESHOLD,
    ).points


@st.cache_data(ttl=300)
def search_image_by_phrase(phrase: str):
    image_vec = get_clip_model().encode(phrase).tolist()
    text_vec = next(iter(get_text_embed_model().embed([phrase]))).tolist()

    image_hits = get_qdrant().query_points(
        collection_name=COLLECTION_NAME, query=image_vec,
        using="image", limit=MAX_RESULTS, with_payload=IMAGE_PAYLOAD,
        score_threshold=IMAGE_SCORE_THRESHOLD,
    ).points

    text_hits = get_qdrant().query_points(
        collection_name=COLLECTION_NAME, query=text_vec,
        using="text", limit=MAX_RESULTS, with_payload=IMAGE_PAYLOAD,
        score_threshold=SCORE_THRESHOLD,
    ).points

    seen = {}
    for hit in image_hits:
        seen[hit.id] = (hit, "image")
    for hit in text_hits:
        if hit.id not in seen:
            seen[hit.id] = (hit, "text")
        else:
            seen[hit.id] = (seen[hit.id][0], "both")  # found in both

    return [(hit, source) for hit, source in seen.values()]



@st.cache_data(ttl=300)
def search_image_by_upload(image_bytes: bytes):
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    vec = get_clip_model().encode(pil_image).tolist()
    return get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        using="image",
        limit=MAX_RESULTS,
        with_payload=IMAGE_PAYLOAD,
        score_threshold=IMAGE_SCORE_THRESHOLD,
    ).points


# ── RAG context (explore page) ─────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_pokemon_context(name: str) -> dict:
    results, _ = get_qdrant().scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="name", match=MatchValue(value=name))]
        ),
        limit=1,
        with_payload=True,
    )
    text_payload = results[0].payload if results else {}
    return {
        "text_payload": text_payload,
        "image_b64": text_payload.get("image_b64"),
        "artwork_url": text_payload.get("artwork_url"),
    }


# ── Tool used by the chat ReAct loop ──────────────────────────────────────────

@tool
def search_pokemon(query: str) -> str:
    """Search the Pokemon database for Pokemon matching the query.

    Use this when the user asks about a Pokemon that is not in the current
    conversation's database record. The query can be a Pokemon name
    (e.g. "charizard") or a description (e.g. "yellow electric mouse").
    Returns up to 3 matches with name, types, abilities, base stats,
    height, weight, and moves.
    """
    # Fetching "data" here (unlike TEXT_PAYLOAD) is intentional — the tool
    # needs the full moves list to give the LLM complete information.
    vec = next(iter(get_text_embed_model().embed([query]))).tolist()
    hits = get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        using="text",
        limit=3,
        with_payload=["name", "types", "abilities", "stats", "height", "weight", "data"],
        score_threshold=SCORE_THRESHOLD,
    ).points

    if not hits:
        return "No Pokemon found in the database for that query."

    parts = []
    for hit in hits:
        p = hit.payload
        data = p.get("data", {})
        moves = [m["move"]["name"].replace("-", " ") for m in data.get("moves", [])]
        stats_str = ", ".join(f"{k}: {v}" for k, v in p.get("stats", {}).items())
        parts.append(
            f"Name: {p.get('name', '').replace('-', ' ').title()}\n"
            f"Types: {', '.join(p.get('types', []))}\n"
            f"Abilities: {', '.join(p.get('abilities', []))}\n"
            f"Stats: {stats_str}\n"
            f"Height: {p.get('height', 0) / 10:.1f}m  Weight: {p.get('weight', 0) / 10:.1f}kg\n"
            f"Moves: {', '.join(moves)}"
        )
    return "\n\n---\n\n".join(parts)
