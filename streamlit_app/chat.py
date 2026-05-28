from langchain_core.messages import AIMessage, ToolMessage

from cache import cache_lookup, cache_store
from clients import get_llm
from search import search_pokemon


def extract_text_content(content):
    """Extract text from response content, handling both string and content-block formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [block.get("text", "") for block in content if block.get("type") == "text"]
        return "".join(texts)
    return str(content)


def run_chat(pokemon_name: str, question: str, lc_messages: list) -> tuple[str, list]:
    """Run one user turn through the cache → LLM → tool loop.

    Flow:
      1. Check Redis for a semantically similar cached answer for this Pokemon.
      2. On a cache hit: return immediately without touching the LLM.
      3. On a cache miss: invoke the LLM with native tool calling. If it
         returns tool_calls, execute each, append a ToolMessage with the
         result, and loop. Otherwise return the response as the final answer.

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
    # bind_tools sends the search_pokemon JSON schema to Ollama on each call so
    # the model can emit a structured tool_call instead of free-form text.
    llm_with_tools = get_llm().bind_tools([search_pokemon])
    messages = list(lc_messages)  # shallow copy so we don't mutate the caller's list

    # Up to 5 iterations: each loop either gets a final answer or executes
    # the tool calls the model requested and feeds the results back.
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        print(f"LLM response: {str(response)}  tool_calls={response.tool_calls}")
        # No tool calls → this is the final answer.
        if not response.tool_calls:
            text_content = extract_text_content(response.content)
            cache_store(pokemon_name, question, text_content)
            return text_content, messages

        # ── Tool execution (ReAct step) ────────────────────────────────────────
        # Run every tool call the model emitted and append a ToolMessage per
        # call. The tool_call_id link is required so the model can match each
        # result to the call it made on the next iteration.
        for tool_call in response.tool_calls:
            result = search_pokemon.invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    # Safety net: if we exhausted all iterations without a clean answer,
    # return whatever the last LLM response was.
    text_content = extract_text_content(response.content)
    return text_content, messages
