import re

from langchain_core.messages import AIMessage, HumanMessage

from cache import cache_lookup, cache_store
from clients import get_llm
from search import get_search_fn

# Regex that detects when the LLM wants to invoke the search tool.
# The model is instructed (in prompts.py) to output a line like:
#   SEARCH_POKEMON: charizard
# This pattern captures everything after the colon as the search query.
_TOOL_RE = re.compile(r"SEARCH_POKEMON:\s*(.+)", re.IGNORECASE)


def run_chat(pokemon_name: str, question: str, lc_messages: list) -> tuple[str, list]:
    """Run one user turn through the cache → LLM → tool loop.

    Flow:
      1. Check Redis for a semantically similar cached answer for this Pokemon.
      2. On a cache hit: return immediately without touching the LLM.
      3. On a cache miss: invoke the LLM, handle any tool calls (ReAct loop),
         cache the final answer, and return it.

    Returns (response_text, updated_lc_messages) so the caller can keep the
    full conversation history for the next turn.
    """
    # ── Cache check ────────────────────────────────────────────────────────────
    cached = cache_lookup(pokemon_name, question)
    if cached:
        # Append the cached answer to the message history so the conversation
        # stays coherent for future turns that might reference this answer.
        messages = list(lc_messages)
        messages.append(AIMessage(content=cached))
        return cached, messages

    # ── LLM invocation ─────────────────────────────────────────────────────────
    llm = get_llm()
    search = get_search_fn()
    messages = list(lc_messages)  # shallow copy so we don't mutate the caller's list

    # Up to 5 iterations: each loop either gets a final answer or executes one
    # tool call and feeds the result back for the next LLM invocation.
    for _ in range(5):
        response = llm.invoke(messages)
        text = response.content

        # Check whether the LLM wants to search for another Pokemon.
        match = _TOOL_RE.search(text)
        if not match:
            # No tool call — this is the final answer.
            messages.append(response)
            # Store in Redis so similar future questions skip the LLM entirely.
            cache_store(pokemon_name, question, text)
            return text, messages

        # ── Tool execution (ReAct step) ────────────────────────────────────────
        query = match.group(1).strip()  # e.g. "charizard"
        result = search(query)          # hit Qdrant and format as plain text

        # Add the LLM's "I want to search" message to history, then inject
        # the search result as a new HumanMessage so the LLM sees it on the
        # next iteration and can answer based on actual database data.
        messages.append(response)
        messages.append(HumanMessage(
            content=f"[Database search result for '{query}']:\n{result}\n\n"
                    "Now answer the user's question using only the above database result."
        ))

    # Safety net: if we exhausted all iterations without a clean answer,
    # return whatever the last LLM response was.
    messages.append(response)
    return response.content, messages
