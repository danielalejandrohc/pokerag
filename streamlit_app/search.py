from io import BytesIO

import streamlit as st
from langchain_core.tools import tool
from PIL import Image
from qdrant_client.models import FieldCondition, Filter, MatchValue

from clients import (
    get_clip_text_model,
    get_image_embed_model,
    get_qdrant,
    get_text_embed_model,
)
from config import (
    COLLECTION_IMAGES,
    COLLECTION_TEXT,
    IMAGE_PAYLOAD,
    MAX_RESULTS,
    TEXT_PAYLOAD,
)


# ── Readiness ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def check_ready() -> bool:
    # Verify both Qdrant collections exist before showing the UI.
    # Cached for 30 s so the readiness check doesn't hit Qdrant on every
    # Streamlit re-run (which happens on every user interaction).
    try:
        existing = {c.name for c in get_qdrant().get_collections().collections}
        return COLLECTION_TEXT in existing and COLLECTION_IMAGES in existing
    except Exception:
        return False


# ── App search (main tabs) ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def search_text(query: str):
    # Embed the user's text query and find the closest Pokemon in the text collection.
    vec = next(iter(get_text_embed_model().embed([query]))).tolist()
    hits = get_qdrant().query_points(
        collection_name=COLLECTION_TEXT,
        query=vec,
        limit=MAX_RESULTS,
        with_payload=TEXT_PAYLOAD,  # only fetch the display fields, not the huge "data" blob
    ).points

    # The text collection doesn't store image_b64, so we do a second lookup
    # against the image collection using the Pokemon IDs from the first results.
    # This is the "join" that lets the text tab show locally stored images.
    pokemon_ids = [h.payload.get("id") for h in hits if h.payload.get("id")]
    if pokemon_ids:
        img_records = get_qdrant().retrieve(
            collection_name=COLLECTION_IMAGES,
            ids=pokemon_ids,
            with_payload=["image_b64"],  # only the image field, nothing else
        )
        # Build an id → base64 lookup so we can attach images in O(1) per hit.
        img_b64_by_id = {r.id: r.payload.get("image_b64") for r in img_records}
        for hit in hits:
            b64 = img_b64_by_id.get(hit.payload.get("id"))
            if b64:
                hit.payload["image_b64"] = b64  # attach to payload for the UI to render

    return hits


@st.cache_data(ttl=300)
def search_image_by_phrase(phrase: str):
    # CLIP text encoder maps the phrase into the same 512-dim space as the
    # image embeddings, so "yellow electric mouse" finds Pikachu artwork
    # without needing a text description of each image.
    vec = next(iter(get_clip_text_model().embed([phrase]))).tolist()
    return get_qdrant().query_points(
        collection_name=COLLECTION_IMAGES,
        query=vec,
        limit=MAX_RESULTS,
        with_payload=IMAGE_PAYLOAD,
    ).points


@st.cache_data(ttl=300)
def search_image_by_upload(image_bytes: bytes):
    # Convert the raw upload bytes to a PIL image (normalises format/colour space),
    # then embed it with the CLIP vision encoder for visual similarity search.
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    vec = next(iter(get_image_embed_model().embed([pil_image]))).tolist()
    return get_qdrant().query_points(
        collection_name=COLLECTION_IMAGES,
        query=vec,
        limit=MAX_RESULTS,
        with_payload=IMAGE_PAYLOAD,
    ).points


# ── RAG context (explore page) ─────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_pokemon_context(name: str) -> dict:
    """Fetch full text payload + image_b64 for a Pokemon from both collections.

    The text collection holds stats, abilities, moves (inside the 'data' field), etc.
    The image collection holds the base64-encoded official artwork.
    Both are returned together so the explore page and the LLM have everything.
    """
    client = get_qdrant()

    # scroll() with a filter is a direct key lookup by payload value — much
    # faster than a vector search when we already know the exact name.
    results, _ = client.scroll(
        collection_name=COLLECTION_TEXT,
        scroll_filter=Filter(
            must=[FieldCondition(key="name", match=MatchValue(value=name))]
        ),
        limit=1,
        with_payload=True,  # fetch ALL fields including the full "data" blob for moves
    )
    text_payload = results[0].payload if results else {}

    image_b64 = None
    artwork_url = text_payload.get("artwork_url")
    pokemon_id = text_payload.get("id")

    # Both collections share the same numeric Pokemon ID as the point ID,
    # so we can retrieve the image record in a single direct lookup.
    if pokemon_id:
        img_records = client.retrieve(
            collection_name=COLLECTION_IMAGES,
            ids=[pokemon_id],
            with_payload=["image_b64", "artwork_url"],
        )
        if img_records:
            image_b64 = img_records[0].payload.get("image_b64")
            artwork_url = img_records[0].payload.get("artwork_url", artwork_url)

    return {"text_payload": text_payload, "image_b64": image_b64, "artwork_url": artwork_url}


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
        collection_name=COLLECTION_TEXT,
        query=vec,
        limit=3,
        with_payload=["name", "types", "abilities", "stats", "height", "weight", "data"],
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
