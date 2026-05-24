# PokéRAG

A Retrieval-Augmented Generation (RAG) application built with Streamlit that lets you search for Pokémon by text description or visual similarity, and chat with an AI assistant that answers questions grounded in real Pokémon data.

---

## Table of Contents

- [What Is This Project?](#what-is-this-project)
- [Why This Project Is Useful to Learn RAG](#why-this-project-is-useful-to-learn-rag)
- [Architecture](#architecture)
- [Topics Covered](#topics-covered)
- [Project Structure](#project-structure)
- [How Each Component Works](#how-each-component-works)
- [NVIDIA GPU Note & Customization Without a GPU](#nvidia-gpu-note--customization-without-a-gpu)
- [Startup Flow](#startup-flow)

---

## What Is This Project?

PokéRAG uses the entire Pokémon catalog (~1,500 Pokémon from PokeAPI) as a knowledge base. It indexes every Pokémon with two types of vector embeddings — one for text descriptions and one for images — and stores them in a vector database. A language model can then be asked questions about any Pokémon and it answers using only that grounded data, not hallucinated knowledge.

The result is a searchable, conversational Pokémon encyclopedia that demonstrates a complete RAG pipeline end-to-end.

---

## Why This Project Is Useful to Learn RAG

RAG is one of the most important patterns in modern AI engineering. This project is an ideal learning vehicle because it is concrete, multi-modal, and covers every stage of the pipeline with real code:

### 1. The Full RAG Pipeline Is Visible End-to-End

Most tutorials show RAG in pseudocode or with toy data. Here every stage is real and inspectable:

- **Ingestion** — `data_loader/pokemon_loader.py` fetches live data from PokeAPI, generates embeddings, and indexes them into Qdrant. You can read every line.
- **Retrieval** — `streamlit_app/search.py` runs similarity searches against the vector database using cosine distance.
- **Augmentation** — `streamlit_app/prompts.py` builds the prompt by injecting the retrieved Pokémon record (stats, types, moves, artwork) into the context window.
- **Generation** — `streamlit_app/chat.py` sends the augmented prompt to the LLM and parses its response.

### 2. Multi-Modal Retrieval

The project shows two distinct embedding strategies side by side:

- **Dense text search** using `BAAI/bge-small-en-v1.5` (BGE, 384-dim) — find Pokémon by description like "fire-breathing dragon with high attack".
- **Visual search** using CLIP (`Qdrant/clip-ViT-B-32-vision` and `Qdrant/clip-ViT-B-32-text`, 512-dim) — find Pokémon by uploading a picture or typing a visual phrase like "round yellow creature".

This teaches when to use general-purpose text embeddings versus vision-language models.

### 3. Tool Use / ReAct Loop

The LLM is not just asked a question and left alone. It can decide to call a search tool (`SEARCH_POKEMON: <query>`) mid-conversation to retrieve related Pokémon before answering. This ReAct-style loop in `chat.py` is a core agentic pattern that every RAG practitioner needs to understand.

### 4. Semantic Caching

`cache.py` implements a semantic cache on top of Redis: instead of hitting the LLM again for questions that are semantically similar to ones already answered, it returns the cached answer. This teaches a real production concern — LLM cost and latency — and how vector similarity applies beyond search.

### 5. Grounding and Prompt Engineering

The system prompt in `prompts.py` explicitly restricts the LLM to the database record. This demonstrates the core promise of RAG: reducing hallucination by grounding the model in retrieved facts rather than its training weights.

### 6. Fully Containerized Infrastructure

The project runs five services (Redis, Qdrant, Ollama, an init service, and the app itself) via a single `docker-compose up`. Learners see how a production AI stack is assembled, not just the model call.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP :8501
┌───────────────────────────▼─────────────────────────────────────┐
│                    Streamlit Application                         │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │   app.py     │   │  explore.py  │   │      chat.py         │ │
│  │  (Search UI) │   │ (Detail/Chat)│   │   (ReAct Loop)       │ │
│  └──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘ │
│         │                  │                       │             │
│  ┌──────▼──────────────────▼───────────────────────▼──────────┐ │
│  │                     search.py / cache.py                   │ │
│  └──────┬───────────────────────────────────────┬─────────────┘ │
└─────────┼───────────────────────────────────────┼───────────────┘
          │                                       │
          │ gRPC :6333                            │ TCP :6379
┌─────────▼──────────┐                 ┌──────────▼──────────┐
│      Qdrant        │                 │        Redis         │
│  (Vector Database) │                 │  (Semantic Cache)    │
│                    │                 │                      │
│  ┌──────────────┐  │                 │  Stores Q&A pairs    │
│  │ pokemons     │  │                 │  with embeddings for │
│  │ (BGE text)   │  │                 │  similarity lookup   │
│  ├──────────────┤  │                 └──────────────────────┘
│  │pokemon_images│  │
│  │ (CLIP vision)│  │
│  └──────────────┘  │
└────────────────────┘
          │
          │ HTTP :11434
┌─────────▼──────────┐
│       Ollama        │
│  (LLM Inference)   │
│  Model: gemma4:e4b  │
└────────────────────┘
```

### Data Flow — Chat Request

```
User question
     │
     ▼
Semantic cache lookup (Redis + cosine similarity)
     │
     ├── Cache HIT ──────────────────────────────► Return cached answer
     │
     └── Cache MISS
              │
              ▼
         Build prompt
         (system prompt + Pokémon record + chat history)
              │
              ▼
         LLM (Ollama / Gemma 4)
              │
              ├── Tool call detected: SEARCH_POKEMON: <query>
              │         │
              │         ▼
              │    Qdrant vector search
              │         │
              │         ▼
              │    Inject results into context
              │         │
              │         └──► LLM continues (up to 5 iterations)
              │
              └── Final text response
                        │
                        ▼
                  Store in Redis cache
                        │
                        ▼
                  Return to user
```

### Data Flow — Ingestion (one-time)

```
PokeAPI (~1,500 Pokémon)
     │
     ▼
pokemon_loader.py  (10 concurrent workers)
     │
     ├── Text path:
     │    Name + types + abilities + stats + moves
     │         │
     │         ▼
     │    BGE embedding (384-dim)
     │         │
     │         ▼
     │    Qdrant "pokemons" collection
     │
     └── Image path:
          Official artwork (PNG)
               │
               ▼
          CLIP vision embedding (512-dim)
          + base64 image stored in payload
               │
               ▼
          Qdrant "pokemon_images" collection
```

---

## Topics Covered

| Topic | Where in the codebase |
|---|---|
| Vector embeddings (dense text) | `clients.py` → `get_text_embed_model()`, `cache.py` |
| Vector embeddings (vision / CLIP) | `clients.py` → `get_image_embed_model()`, `get_clip_text_model()` |
| Vector database (Qdrant) | `data_loader/pokemon_loader.py`, `search.py` |
| Similarity search (cosine) | `search.py` → all search functions |
| Multi-modal search | `app.py` image tab vs text tab |
| RAG prompt construction | `prompts.py` → `build_system_prompt()` |
| Grounded generation | `chat.py` → system prompt restricts LLM to DB record |
| ReAct tool-use loop | `chat.py` → `run_chat()` SEARCH_POKEMON intercept |
| Semantic caching | `cache.py` → Redis-backed cosine similarity cache |
| Containerized AI stack | `docker-compose.yml` (5 services) |
| Data ingestion pipeline | `data_loader/pokemon_loader.py` |
| Conversation state management | `explore.py` → `st.session_state` per Pokémon |
| LangChain integration | `clients.py` → `get_llm()` (ChatOllama) |
| Multimodal prompting | `prompts.py` → base64 image in HumanMessage |

---

## Project Structure

```
pokerag/
├── data_loader/
│   └── pokemon_loader.py       # One-time ingestion: fetches PokeAPI, embeds, indexes
├── streamlit_app/
│   ├── app.py                  # Main page: text search + image search tabs
│   ├── pages/
│   │   └── explore.py          # Detail page: Pokémon card + RAG chat interface
│   ├── cache.py                # Semantic cache using Redis + cosine similarity
│   ├── chat.py                 # ReAct loop: LLM + SEARCH_POKEMON tool
│   ├── clients.py              # Singleton clients: Qdrant, Redis, LLM, embedding models
│   ├── config.py               # All constants: hosts, ports, model names, thresholds
│   ├── prompts.py              # System prompt builder and initial message seeding
│   ├── search.py               # Vector search helpers + context fetcher
│   └── ui.py                   # Pokémon card and grid rendering
├── Dockerfile
├── docker-compose.yml          # Full stack: Redis, Qdrant, Ollama, init, app
└── requirements.txt
```

---

## How Each Component Works

### `data_loader/pokemon_loader.py` — Ingestion

Runs once at startup (skips if collections already exist). Uses 10 async workers to fetch all ~1,500 Pokémon from the public PokeAPI. For each Pokémon it builds a text description (name, types, abilities, base stats, first 20 moves) and embeds it with BGE. Separately it downloads the official artwork PNG, embeds it with CLIP, and stores the base64-encoded image in the Qdrant payload so the app never needs to re-fetch it.

### `streamlit_app/search.py` — Retrieval

Four search functions, all returning scored Qdrant hits:

- `search_text(query)` — embed query with BGE → search `pokemons` collection
- `search_image_by_phrase(phrase)` — embed phrase with CLIP text encoder → search `pokemon_images`
- `search_image_by_upload(bytes)` — embed image bytes with CLIP vision encoder → search `pokemon_images`
- `fetch_pokemon_context(name)` — retrieve the full record for a named Pokémon from both collections (used to build the RAG prompt)

### `streamlit_app/chat.py` — Generation + Tool Use

The `run_chat()` function drives the full RAG loop:

1. Check the semantic cache. Return immediately on a hit.
2. Build the prompt (system message with the full Pokémon record + chat history).
3. Call the LLM. Scan the response for the pattern `SEARCH_POKEMON: <query>`.
4. If the pattern is found, run the search tool, append the results to the context, and call the LLM again. Repeat up to 5 times.
5. On a final answer, store it in the semantic cache and return.

### `streamlit_app/cache.py` — Semantic Cache

On every question, the cache embeds the question with BGE, then scans all Redis hashes stored under the current Pokémon's namespace. If the cosine similarity between the new question and any cached question exceeds `SIM_THRESHOLD = 0.88`, the stored answer is returned without calling the LLM. Entries expire after one hour.

### `streamlit_app/prompts.py` — Grounding

The system prompt tells the LLM: you are a Pokémon database assistant; answer only from the provided record; do not use outside knowledge. The record injected includes types, abilities, stats, moves, and the base64 artwork. This is the grounding step that makes the generation trustworthy.

---

## NVIDIA GPU Note & Customization Without a GPU

### Why NVIDIA Is Required by Default

The `docker-compose.yml` file configures the Ollama service to use all available NVIDIA GPUs:

```yaml
ollama:
  image: ollama/ollama
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

This is needed because the default model — `gemma4:e4b` — is a large language model that benefits significantly from GPU acceleration. Running it on CPU is possible but very slow (minutes per response instead of seconds).

You also need the **NVIDIA Container Toolkit** installed on your host so Docker can pass GPU access into the container.

### Option 1 — Run Without a GPU (CPU Mode)

If you do not have an NVIDIA GPU, remove the `deploy` block from the `ollama` service in `docker-compose.yml`:

```yaml
# Before (GPU required):
ollama:
  image: ollama/ollama
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

# After (CPU only):
ollama:
  image: ollama/ollama
```

Then switch to a smaller, faster-on-CPU model. Open `streamlit_app/config.py` and change:

```python
# Before:
OLLAMA_MODEL = "gemma4:e4b"

# After (pick one):
OLLAMA_MODEL = "llama3.2:1b"   # Very fast on CPU, less capable
OLLAMA_MODEL = "phi3:mini"      # Good balance of size and quality
OLLAMA_MODEL = "gemma3:1b"      # Compact Gemma variant
```

Update the model pulled in `docker-compose.yml` to match:

```yaml
ollama-init:
  command: >
    sh -c "
      until ollama list; do sleep 2; done &&
      ollama pull llama3.2:1b   # change this to match config.py
    "
```

CPU inference will work, but responses may take 30–120 seconds depending on your hardware and model choice.

### Option 2 — Use AMD GPU (ROCm)

Replace the Ollama image with the ROCm variant and expose the GPU device:

```yaml
ollama:
  image: ollama/ollama:rocm
  devices:
    - /dev/kfd
    - /dev/dri
```

### Option 3 — Use a Remote or Cloud LLM Instead of Ollama

If you want faster responses without local hardware, you can replace the Ollama backend with the OpenAI-compatible API provided by services like OpenRouter, Groq, or Together AI. Swap `ChatOllama` for `ChatOpenAI` (which works with any OpenAI-compatible endpoint) in `streamlit_app/clients.py`:

```python
# Before (Ollama):
from langchain_ollama import ChatOllama
def get_llm():
    return ChatOllama(base_url=config.OLLAMA_HOST, model=config.OLLAMA_MODEL, temperature=0.1)

# After (any OpenAI-compatible API):
from langchain_openai import ChatOpenAI
def get_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",   # or Groq, Together, etc.
        api_key="your-api-key",
        model="meta-llama/llama-3.1-8b-instruct",
        temperature=0.1,
    )
```

You can then remove the `ollama` and `ollama-init` services from `docker-compose.yml` entirely.

> **Note:** The text and image embedding models (`fastembed`) run entirely on CPU and require no GPU. Only the LLM inference (Ollama) benefits from GPU acceleration.

---

## Startup Flow

```
docker-compose up
       │
       ├─► Redis starts          (port 6379)
       ├─► Qdrant starts         (port 6333)
       └─► Ollama starts         (port 11434)
                │
                ▼
         ollama-init waits for Ollama health check
                │
                ▼
         ollama pull gemma4:e4b  (downloads model weights)
                │
                ▼
         app service starts
                │
                ├── python data_loader/pokemon_loader.py
                │       Fetches ~1,500 Pokémon, embeds, indexes into Qdrant
                │       (skipped on subsequent runs if collections exist)
                │
                └── streamlit run streamlit_app/app.py
                        │
                        ▼
                 http://localhost:8501
```

On first run the ingestion step takes several minutes. On subsequent runs it is skipped and the app starts in seconds.
