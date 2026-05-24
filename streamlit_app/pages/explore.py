import base64

import streamlit as st
from langchain_core.messages import HumanMessage

from chat import run_chat
from prompts import build_initial_lc_messages
from search import fetch_pokemon_context

st.set_page_config(page_title="PokéRAG — Explore", page_icon="🔍", layout="wide")

# Read the Pokemon name that was stored in session_state by ui.render_card
# when the user clicked the "Explore" button.
pokemon_name = st.session_state.get("selected_pokemon", "")

if not pokemon_name:
    # This page should only ever be reached via the Explore button, not directly.
    st.error("No Pokemon selected.")
    if st.button("← Back to Search"):
        st.switch_page("app.py")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
# fetch_pokemon_context is cached for 1 hour so navigating back and forth
# between Pokemon does not re-query Qdrant every time.
with st.spinner("Loading Pokemon data…"):
    context = fetch_pokemon_context(pokemon_name)

payload = context["text_payload"]
# Convert internal name format ("mr-mime") to display format ("Mr Mime").
display_name = pokemon_name.replace("-", " ").title()

col_img, col_info = st.columns([1, 3])

with col_img:
    if context["image_b64"]:
        # Decode the base64 string stored in Qdrant back to raw image bytes
        # that st.image can render directly without any HTTP request.
        st.image(base64.b64decode(context["image_b64"]), use_container_width=True)
    elif context["artwork_url"]:
        # Fallback for Pokemon whose image was not indexed (e.g. load failed).
        st.image(context["artwork_url"], use_container_width=True)

with col_info:
    st.title(f"Explore {display_name}")
    types = payload.get("types", [])
    if types:
        st.write(" · ".join(f"`{t}`" for t in types))
    stats = payload.get("stats", {})
    if stats:
        keys = ["hp", "attack", "defense", "speed"]
        st.caption("  ".join(f"{k}: {stats[k]}" for k in keys if k in stats))
    h, w = payload.get("height"), payload.get("weight")
    if h and w:
        st.caption(f"{h / 10:.1f} m · {w / 10:.1f} kg")
    st.caption("Ask anything about this Pokemon. The assistant answers only from the database.")

if st.button("← Back to Search"):
    st.switch_page("app.py")

st.divider()

# ── Conversation state ─────────────────────────────────────────────────────────
# Each Pokemon gets its own key in session_state so switching between Pokemon
# preserves all conversations independently.
state_key = f"explore_{pokemon_name}"
if state_key not in st.session_state:
    st.session_state[state_key] = {
        "display": [],  # plain dicts shown in the chat UI: {"role": ..., "content": ...}
        "lc": None,     # list[BaseMessage] passed to the LLM; None until the first turn
    }

conv = st.session_state[state_key]

# Re-render the full conversation history on every Streamlit re-run so the
# chat UI stays populated when the user interacts with other widgets.
for msg in conv["display"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input(f"Ask about {display_name}…")

if user_input:
    # Show the user's message immediately, before the LLM responds.
    conv["display"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # First turn: initialise the LangChain message list with the system prompt
    # and (if available) the Pokemon's official artwork as a multimodal message.
    if conv["lc"] is None:
        conv["lc"] = build_initial_lc_messages(context)

    # Append the user's question to the LangChain history before passing it to
    # run_chat, so the LLM sees the full conversation context including this turn.
    conv["lc"].append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response_text, updated_lc = run_chat(pokemon_name, user_input, conv["lc"])
            except Exception as exc:
                response_text = f"Error communicating with the model: {exc}"
                updated_lc = conv["lc"]
        st.markdown(response_text)

    # Persist the updated message list (which now includes the assistant's reply)
    # so the next turn has the full conversation history.
    conv["lc"] = updated_lc
    conv["display"].append({"role": "assistant", "content": response_text})
