import logging
import streamlit as st

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
for _noisy in ("httpx", "httpcore", "filelock", "PIL", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from search import check_ready, search_text, search_image_by_phrase
from ui import render_grid

st.set_page_config(page_title="PokéRAG", page_icon="🔴", layout="wide")
st.title("PokéRAG — Pokémon Search")

# Block the UI until both Qdrant collections are ready.
# The data_loader runs before Streamlit starts, but on the first boot it can
# take a few minutes to index all Pokemon — this gives the user clear feedback.
if not check_ready():
    st.info("⏳ Qdrant is loading — data is still being indexed. Check back in a moment.")
    if st.button("Refresh"):
        st.rerun()
    st.stop()

tab_txt = st.tabs(["🔤 Search by Text"])[0]

# ── Text tab ───────────────────────────────────────────────────────────────────
with tab_txt:
    st.subheader("Find Pokémon by description")
    
    col_search, col_filter = st.columns([3, 1])

    with col_search:
        query = st.text_input(
            "Describe the Pokémon you're looking for",
            placeholder="A small orange dragon-like lizard with large blue eyes, a rounded snout, a pale cream belly, and a tail ending in a flickering fire",
            key="txt_query",
        )
    with col_filter:
        source_filter = st.selectbox(
            "Filter by Source",
            options=["All", "Text", "Image"],
            index=0,
            help="Choose whether to show Pokémon found via text description, visual similarity, or both."
        )

    if query:
        with st.spinner("Searching…"):
            hits = search_image_by_phrase(query)

        # Apply the sidebar filter to the search results
        if source_filter == "Text":
            hits = [h for h in hits if h[1] in ("text", "both")]
        elif source_filter == "Image":
            hits = [h for h in hits if h[1] in ("image", "both")]

        st.write(f"**{len(hits)} results** sorted by text similarity")
        render_grid(hits, use_b64=True)
