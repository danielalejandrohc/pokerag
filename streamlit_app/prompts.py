from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def build_system_prompt(text_payload: dict) -> str:
    # Pull every field the LLM will need from the Qdrant payload.
    data = text_payload.get("data", {})  # raw PokeAPI JSON — needed for the full moves list
    name = text_payload.get("name", "unknown").replace("-", " ").title()
    types = ", ".join(text_payload.get("types", []))
    abilities = ", ".join(text_payload.get("abilities", []))
    stats = text_payload.get("stats", {})

    # Qdrant stores height/weight as decimetres/hectograms (PokeAPI convention).
    # Divide by 10 to convert to metres / kilograms for human-readable output.
    height_m = text_payload.get("height", 0) / 10
    weight_kg = text_payload.get("weight", 0) / 10

    # Flatten the nested moves list from the raw PokeAPI structure.
    moves = [m["move"]["name"].replace("-", " ") for m in data.get("moves", [])] if data else []

    # Format stats as indented lines so the LLM can read them clearly.
    stats_lines = "\n".join(f"  {k}: {v}" for k, v in stats.items())
    moves_text = ", ".join(moves) if moves else "none recorded"

    # The system prompt does three things:
    #   1. Restricts the LLM to database knowledge only (no hallucinated Pokedex entries).
    #   2. Tells it how to handle visual questions (use the artwork, not its training data).
    #   3. Points the model at the `search_pokemon` tool for lookups of other Pokemon.
    return f"""You are a Pokemon database assistant. Your ONLY source of knowledge about Pokemon \
is the database record below. Do NOT use any knowledge from your training data about Pokemon.

If the user asks about visual appearance, base your answer solely on the official artwork image \
provided at conversation start and the physical attributes listed below.

If the user asks about other Pokemon not in the record below, you MUST call the `search_pokemon` \
tool to look them up first. Never answer questions about other Pokemon from memory.

If information is not in the database, say "That information is not in the database."

=== DATABASE RECORD: {name} ===
Types: {types}
Abilities: {abilities}
Height: {height_m:.1f} m
Weight: {weight_kg:.1f} kg
Base Stats:
{stats_lines}
Moves: {moves_text}
================================
"""


def build_initial_lc_messages(context: dict) -> list:
    """Build the opening LangChain message list for a new conversation.

    Always starts with a SystemMessage containing the database record.
    If an image is available, a Human+AI exchange is prepended so the model
    'sees' the official artwork before the user's first question — enabling
    accurate answers to visual / appearance questions.
    """
    messages = [SystemMessage(content=build_system_prompt(context["text_payload"]))]

    if context.get("image_b64"):
        # Send the image as a multimodal HumanMessage.
        # The data-URI format (data:image/png;base64,…) is what ChatOllama
        # expects for inline images — it never makes an outbound HTTP request.
        messages.append(HumanMessage(content=[
            {"type": "text", "text": "Here is the official artwork stored in the database for this Pokemon:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{context['image_b64']}"}},
        ]))
        # Seed a synthetic AI acknowledgement so the model knows it has already
        # processed the image and won't re-describe it unprompted.
        messages.append(AIMessage(
            content="I can see the Pokemon's official artwork from the database. "
                    "I'm ready to answer questions based solely on the database records."
        ))

    return messages
