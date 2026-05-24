import base64

import streamlit as st

from config import N_COLS


def render_card(col, payload: dict, score: float, use_b64: bool = False) -> None:
    """Render a single Pokemon result card inside the given Streamlit column."""
    with col:
        # Prefer the locally stored base64 image (from the image collection)
        # over the remote artwork URL to avoid external HTTP requests.
        if use_b64 and payload.get("image_b64"):
            st.image(base64.b64decode(payload["image_b64"]), use_container_width=True)
        elif payload.get("artwork_url"):
            # Fallback: load the image directly from the PokeAPI CDN URL.
            st.image(payload["artwork_url"], use_container_width=True)

        # Normalise the name: "mr-mime" → "Mr Mime"
        name = payload.get("name", "?").replace("-", " ").title()
        st.markdown(f"**{name}**")

        # Show the similarity score as a percentage (e.g. "87.3%").
        st.caption(f"Similarity: {score:.1%}")

        types = payload.get("types", [])
        if types:
            # Render each type as inline code for a badge-like appearance.
            st.write(" · ".join(f"`{t}`" for t in types))

        stats = payload.get("stats", {})
        if stats:
            # Show the four most battle-relevant stats in a compact single line.
            keys = ["hp", "attack", "defense", "speed"]
            st.caption("  ".join(f"{k}: {stats[k]}" for k in keys if k in stats))

        h = payload.get("height")
        w = payload.get("weight")
        if h and w:
            # Qdrant stores height in decimetres and weight in hectograms (PokeAPI units).
            st.caption(f"{h / 10:.1f} m · {w / 10:.1f} kg")

        with st.expander("View record"):
            # Exclude image_b64 from the JSON viewer — it's a huge string that
            # would make the expander unreadable and slow to render.
            st.json({k: v for k, v in payload.items() if k != "image_b64"})

        pokemon_name = payload.get("name", "")
        # card_id must be unique within the page because Streamlit requires
        # every widget key to be unique. Use the numeric ID when available,
        # fall back to pokemon_id (image collection field) or the display name.
        card_id = payload.get("id") or payload.get("pokemon_id") or name
        if pokemon_name and st.button("Explore", key=f"explore_{card_id}"):
            # Pass the Pokemon name to the explore page via session_state.
            # Query params are not used because st.switch_page resets the URL.
            st.session_state["selected_pokemon"] = pokemon_name
            st.switch_page("pages/explore.py")


def render_grid(hits, use_b64: bool = False) -> None:
    """Render a list of Qdrant hits as an N_COLS-wide grid of cards."""
    # Slice the hits into rows of N_COLS, creating a new set of columns per row.
    for i in range(0, len(hits), N_COLS):
        cols = st.columns(N_COLS)
        for col, hit in zip(cols, hits[i: i + N_COLS]):
            render_card(col, hit.payload, hit.score, use_b64)
