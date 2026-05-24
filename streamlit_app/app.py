import logging
import streamlit as st

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
for _noisy in ("httpx", "httpcore", "filelock", "PIL", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from search import check_ready, search_image_by_phrase, search_image_by_upload, search_text
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

tab_img, tab_txt = st.tabs(["🖼️ Search by Image", "🔤 Search by Text"])

# ── Image tab ──────────────────────────────────────────────────────────────────
with tab_img:
    st.subheader("Find Pokémon by visual similarity")

    # Two input methods: upload an actual image, or describe it in words.
    # If both are provided, the upload takes precedence (the elif below).
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    img_phrase = st.text_input(
        "…or describe what it looks like",
        placeholder='e.g. "horse", "blue serpent", "round yellow creature"',
        key="img_phrase",
    )

    if uploaded:
        # Read the file bytes once — st.file_uploader resets on every re-run
        # so we must read() before any other Streamlit call.
        image_bytes = uploaded.read()
        st.image(image_bytes, caption="Query image", width=220)
        with st.spinner("Embedding and searching…"):
            hits = search_image_by_upload(image_bytes)
        st.write(f"**{len(hits)} results** sorted by visual similarity")
        render_grid(hits, use_b64=True)
    elif img_phrase:
        with st.spinner("Searching by phrase…"):
            hits = search_image_by_phrase(img_phrase)
        st.write(f"**{len(hits)} results** sorted by visual similarity")
        render_grid(hits, use_b64=True)

# ── Text tab ───────────────────────────────────────────────────────────────────
with tab_txt:
    st.subheader("Find Pokémon by description")
    query = st.text_input(
        "Describe the Pokémon you're looking for",
        placeholder="fire breathing dragon with high attack",
        key="txt_query",
    )
    if query:
        with st.spinner("Searching…"):
            hits = search_text(query)
        st.write(f"**{len(hits)} results** sorted by text similarity")
        render_grid(hits, use_b64=True)
