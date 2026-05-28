# PokéRAG

A Retrieval-Augmented Generation (RAG) application built with Streamlit that lets you search for Pokémon by text description and chat with an AI assistant that answers questions grounded in real Pokémon data.

---

## Table of Contents

- [What Is This Project?](#what-is-this-project)
- [Why This Project Is Useful to Learn RAG](#why-this-project-is-useful-to-learn-rag)
- [Deployment Modes](#deployment-modes)
- [Architecture — Local](#architecture--local)
- [Architecture — Cloud](#architecture--cloud)
- [Data Flows](#data-flows)
- [Topics Covered](#topics-covered)
- [Project Structure](#project-structure)
- [How Each Component Works](#how-each-component-works)
- [Configuration Reference](#configuration-reference)
- [Startup Flow](#startup-flow)

---

## What Is This Project?

PokéRAG uses the entire Pokémon catalog (~1,500 Pokémon from PokeAPI) as a knowledge base. It indexes every Pokémon with text vector embeddings and stores them in a vector database. A language model can then be asked questions about any Pokémon and it answers using only that grounded data, not hallucinated knowledge.

The result is a searchable, conversational Pokémon encyclopedia that demonstrates a complete RAG pipeline end-to-end. It runs equally well with a fully local stack (Ollama + local Qdrant + local Redis) or with cloud-managed services (Gemini + Qdrant Cloud + Redis Cloud) — the same application code handles both, controlled entirely by environment variables.

---

## Why This Project Is Useful to Learn RAG

RAG is one of the most important patterns in modern AI engineering. This project is an ideal learning vehicle because it is concrete and covers every stage of the pipeline with real code:

### 1. The Full RAG Pipeline Is Visible End-to-End

Most tutorials show RAG in pseudocode or with toy data. Here every stage is real and inspectable:

- **Ingestion** — `data_loader/pokemon_loader.py` fetches live data from PokeAPI, generates embeddings, and indexes them into Qdrant. You can read every line.
- **Retrieval** — `streamlit_app/search.py` runs similarity searches against the vector database using cosine distance.
- **Augmentation** — `streamlit_app/prompts.py` builds the prompt by injecting the retrieved Pokémon record (stats, types, moves, artwork) into the context window.
- **Generation** — `streamlit_app/chat.py` sends the augmented prompt to the LLM and parses its response.

### 2. Tool Use / ReAct Loop with Native Tool Calling

The LLM is not just asked a question and left alone. It can decide to call a `search_pokemon` tool mid-conversation to retrieve related Pokémon before answering. The tool is registered using LangChain's native `bind_tools` API, so the model emits a structured `tool_calls` response rather than a text pattern. The ReAct loop in `chat.py` executes each tool call, appends a `ToolMessage` with the result, and loops up to 5 times until the model produces a final answer.

### 3. Semantic Caching

`cache.py` implements a semantic cache on top of Redis using `redisvl`'s `SemanticCache`. Instead of hitting the LLM again for questions that are semantically similar to ones already answered, it returns the cached answer. The cache uses the `redis/langcache-embed-v1` model — fine-tuned for question paraphrase detection — so "who can Pikachu beat?" and "what pokemon can pikachu defeat?" both hit the same cache entry. This teaches a real production concern: LLM cost and latency.

### 4. Grounding and Prompt Engineering

The system prompt in `prompts.py` explicitly restricts the LLM to the database record. This demonstrates the core promise of RAG: reducing hallucination by grounding the model in retrieved facts rather than its training weights.

### 5. LLM Portability

The app selects the LLM backend at runtime based on environment variables. When `GOOGLE_API_KEY` is set it uses `ChatGoogleGenerativeAI` (Gemini). Otherwise it falls back to `ChatOllama` (local Ollama). The rest of the codebase is completely unaware of which backend is running — a clean abstraction that shows how to write LLM-portable applications.

### 6. Fully Containerized Infrastructure (Local Mode)

The project runs four services (Redis Stack, Qdrant, Ollama, and the app) via a single `docker-compose up`. Learners see how a production AI stack is assembled, not just the model call.

---

## Deployment Modes

| | Local | Cloud |
|---|---|---|
| **LLM** | Ollama (`gemma4:e4b`) | Google Gemini (`gemini-2.5-flash`) |
| **Vector DB** | Qdrant container (`:6333`) | Qdrant Cloud (HTTPS) |
| **Semantic Cache** | Redis Stack container (`:6379`) | Redis Cloud (HTTPS) |
| **Embeddings** | fastembed — runs locally in both modes | fastembed — runs locally in both modes |
| **Config trigger** | `GOOGLE_API_KEY` absent | `GOOGLE_API_KEY` present |

Switching between modes requires only changing environment variables — no code changes.

---

## Architecture — Local

All services run as Docker containers on your machine.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP :8501
┌───────────────────────────▼─────────────────────────────────────┐
│                    Streamlit Application                         │
│                                                                 │
│  ┌──────────────────┐     ┌────────────────────────────────────┐ │
│  │     app.py       │     │       pages/explore.py             │ │
│  │  (Text Search)   │     │  (Detail card + RAG chat)          │ │
│  └────────┬─────────┘     └──────────────┬─────────────────────┘ │
│           │                              │                       │
│  ┌────────▼──────────────────────────────▼─────────────────────┐ │
│  │         search.py / cache.py / chat.py / clients.py          │ │
│  └────────┬──────────────────────────────┬─────────────────────┘ │
└───────────┼──────────────────────────────┼───────────────────────┘
            │                              │
            │ gRPC :6333                   │ TCP :6379
┌───────────▼─────────────┐   ┌───────────▼──────────────────────┐
│         Qdrant           │   │          Redis Stack              │
│   (Vector Database)      │   │        (Semantic Cache)           │
│                          │   │                                   │
│  ┌────────────────────┐  │   │  SemanticCache (redisvl)          │
│  │     pokemons       │  │   │  HFTextVectorizer                 │
│  │  BGE text  384-dim │  │   │  model: redis/langcache-embed-v1  │
│  ├────────────────────┤  │   └───────────────────────────────────┘
│  │  pokemon_images    │  │
│  │  CLIP vision 512-d │  │
│  └────────────────────┘  │
└──────────────────────────┘
            │
            │ HTTP :11434
┌───────────▼──────────────┐
│          Ollama           │
│      (Local LLM)          │
│   Model: gemma4:e4b       │
│   GPU: NVIDIA (optional)  │
└──────────────────────────-┘
```

---

## Architecture — Cloud

Only the Streamlit app and fastembed models run locally. All stateful services are cloud-managed.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP :8501
┌───────────────────────────▼─────────────────────────────────────┐
│                    Streamlit Application                         │
│                                                                 │
│  ┌──────────────────┐     ┌────────────────────────────────────┐ │
│  │     app.py       │     │       pages/explore.py             │ │
│  │  (Text Search)   │     │  (Detail card + RAG chat)          │ │
│  └────────┬─────────┘     └──────────────┬─────────────────────┘ │
│           │                              │                       │
│  ┌────────▼──────────────────────────────▼─────────────────────┐ │
│  │         search.py / cache.py / chat.py / clients.py          │ │
│  └────────┬──────────────────┬───────────────────┬─────────────┘ │
│           │                  │                   │               │
│    fastembed (local)         │                   │               │
│    BGE + CLIP embeddings     │                   │               │
└─────────────────────────────┼───────────────────┼───────────────┘
                              │                   │
              HTTPS           │           HTTPS   │         HTTPS
     ┌────────▼──────────┐    │    ┌──────▼──────┐ │   ┌────▼──────────────┐
     │   Qdrant Cloud    │    │    │ Redis Cloud  │ │   │  Google Gemini    │
     │                   │    │    │              │ │   │  (Cloud LLM)      │
     │  QDRANT_API_KEY   │    │    │  REDIS_URL   │ │   │  gemini-2.5-flash │
     │  QDRANT_HOST      │    │    │  REDIS_PASSWORD    │  GOOGLE_API_KEY   │
     └───────────────────┘         └──────────────┘    └───────────────────┘
```

---

## Data Flows

### Chat Request

```
User question
     │
     ▼
Semantic cache lookup (Redis SemanticCache — cosine similarity on question embedding)
     │
     ├── Cache HIT ──────────────────────────────► Return cached answer
     │
     └── Cache MISS
              │
              ▼
         Build LangChain messages
         (SystemMessage with Pokémon record + image + chat history)
              │
              ▼
         LLM.bind_tools([search_pokemon]).invoke(messages)
         (Ollama locally  OR  Gemini in cloud)
              │
              ├── tool_calls present: search_pokemon(query)
              │         │
              │         ▼
              │    Qdrant vector search → top 3 Pokémon
              │         │
              │         ▼
              │    Append ToolMessage with results
              │         │
              │         └──► LLM continues (up to 5 iterations)
              │
              └── No tool_calls → final text response
                        │
                        ▼
                  Store in Redis semantic cache
                        │
                        ▼
                  Return to user
```

### Ingestion (one-time, runs at startup)

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
     │    BGE embedding  (BAAI/bge-small-en-v1.5, 384-dim)
     │         │
     │         ▼
     │    Qdrant "pokemons" collection
     │
     └── Image path:
          Official artwork PNG (from PokeAPI CDN)
               │
               ▼
          CLIP vision embedding (Qdrant/clip-ViT-B-32-vision, 512-dim)
          + base64 image stored in payload
               │
               ▼
          Qdrant "pokemon_images" collection
```

---

## Topics Covered

| Topic | Where in the codebase |
|---|---|
| Vector embeddings (dense text) | `clients.py` → `get_text_embed_model()`, `data_loader/pokemon_loader.py` |
| Vector embeddings (vision / CLIP) | `clients.py` → `get_image_embed_model()`, `get_clip_text_model()` |
| Vector database (Qdrant) | `data_loader/pokemon_loader.py`, `search.py` |
| Similarity search (cosine) | `search.py` → all search functions |
| RAG prompt construction | `prompts.py` → `build_system_prompt()` |
| Grounded generation | `chat.py` → system prompt restricts LLM to DB record |
| Native LangChain tool use | `search.py` → `@tool search_pokemon`, `chat.py` → `bind_tools` + `tool_calls` |
| ReAct loop | `chat.py` → `run_chat()` iterates until no more tool calls |
| Semantic caching (redisvl) | `cache.py` → `SemanticCache` + `HFTextVectorizer` |
| LLM portability (local vs cloud) | `clients.py` → `get_llm()` selects Ollama or Gemini from env |
| Containerized AI stack (local) | `docker-compose.yml` (4 services) |
| Cloud-managed services | `.env` — Qdrant Cloud, Redis Cloud, Google Gemini |
| Data ingestion pipeline | `data_loader/pokemon_loader.py` |
| Conversation state management | `pages/explore.py` → `st.session_state` per Pokémon |
| Multimodal prompting (image in context) | `prompts.py` → `build_initial_lc_messages()` — base64 image as HumanMessage |

---

## Project Structure

```
pokerag/
├── data_loader/
│   └── pokemon_loader.py       # One-time ingestion: fetches PokeAPI, embeds, indexes
├── streamlit_app/
│   ├── app.py                  # Main page: text search
│   ├── pages/
│   │   └── explore.py          # Detail page: Pokémon card + RAG chat interface
│   ├── cache.py                # Semantic cache using Redis SemanticCache (redisvl)
│   ├── chat.py                 # ReAct loop: LLM + native tool calling
│   ├── clients.py              # Singleton clients: Qdrant, Redis, LLM, embedding models
│   ├── config.py               # All constants: hosts, ports, model names, thresholds
│   ├── prompts.py              # System prompt builder and initial message seeding
│   ├── search.py               # Vector search helpers + @tool search_pokemon
│   └── ui.py                   # Pokémon card and grid rendering
├── .env                        # Cloud credentials (Qdrant Cloud, Redis Cloud, Gemini)
├── .env-local                  # Local Docker Compose hostnames (override for local dev)
├── Dockerfile
├── docker-compose.yml          # Full local stack: Redis Stack, Qdrant, Ollama, app
└── requirements.txt
```

---

## How Each Component Works

### `data_loader/pokemon_loader.py` — Ingestion

Runs once at startup (skips if collections already exist). Uses 10 concurrent workers to fetch all ~1,500 Pokémon from the public PokeAPI. For each Pokémon it builds a text description (name, types, abilities, base stats, moves) and embeds it with BGE. Separately it downloads the official artwork PNG, embeds it with CLIP, and stores the base64-encoded image in the Qdrant payload so the app never needs to re-fetch it.

Connects to Qdrant using `QDRANT_API_KEY` if set (cloud) or unauthenticated if not (local).

### `streamlit_app/search.py` — Retrieval + Tool

Three search functions for the UI, all returning scored Qdrant hits:

- `search_text(query)` — embed query with BGE → search `pokemons` collection, then join `pokemon_images` for base64 images
- `search_image_by_phrase(phrase)` — embed phrase with CLIP text encoder → search `pokemon_images`
- `search_image_by_upload(bytes)` — embed image bytes with CLIP vision encoder → search `pokemon_images`

One tool for the LLM:

- `search_pokemon(query)` — decorated with `@tool`, registered with the LLM via `bind_tools`. Returns the top 3 Pokémon with full stats and moves as a formatted string.

### `streamlit_app/chat.py` — Generation + Tool Use

The `run_chat()` function drives the full RAG loop:

1. Check the semantic cache. Return immediately on a hit.
2. Invoke `llm.bind_tools([search_pokemon])` with the full message history.
3. If the model returns `tool_calls`, execute each call, append a `ToolMessage`, and call the LLM again. Repeat up to 5 iterations.
4. When the model returns a response with no tool calls, store it in the semantic cache and return.

### `streamlit_app/cache.py` — Semantic Cache

Uses `redisvl`'s `SemanticCache` with `HFTextVectorizer` (`redis/langcache-embed-v1`). On every question the cache embeds the question and checks Redis for semantically similar entries. If the cosine distance is within `CACHE_DISTANCE_THRESHOLD = 0.135` (≈ 86.5% similarity), the stored answer is returned without calling the LLM. When `REDIS_PASSWORD` is set, connects with authentication (cloud mode).

### `streamlit_app/clients.py` — Client Singletons

All clients are `@st.cache_resource` — created once per Streamlit server process:

- `get_qdrant()` — uses `QDRANT_API_KEY` for cloud, unauthenticated for local
- `get_semantic_cache()` — uses `REDIS_PASSWORD`/`REDIS_USERNAME` for cloud, no auth for local
- `get_llm()` — returns `ChatGoogleGenerativeAI` when `GOOGLE_API_KEY` is set, `ChatOllama` otherwise
- `get_text_embed_model()` — BGE model, always runs locally via fastembed
- `get_image_embed_model()` — CLIP vision encoder, always runs locally via fastembed
- `get_clip_text_model()` — CLIP text encoder, always runs locally via fastembed

### `streamlit_app/prompts.py` — Grounding

The system prompt tells the LLM: you are a Pokémon database assistant; answer only from the provided record; do not use outside knowledge. The record injected includes types, abilities, stats, moves. If an image is available, it is sent as a multimodal `HumanMessage` (base64 data URI) at the start of the conversation so the model can answer visual appearance questions accurately.

---

## Configuration Reference

### Environment Variables

| Variable | Local default | Cloud value |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant Cloud hostname |
| `QDRANT_PORT` | `6333` | `6333` |
| `QDRANT_API_KEY` | _(unset)_ | Qdrant Cloud API key |
| `REDIS_URL` | `redis://localhost:6379` | Redis Cloud URL |
| `REDIS_USERNAME` | `default` | Redis Cloud username |
| `REDIS_PASSWORD` | _(unset)_ | Redis Cloud password |
| `OLLAMA_HOST` | `http://localhost:11434` | _(not used in cloud mode)_ |
| `OLLAMA_MODEL` | `gemma4:e4b` | _(not used in cloud mode)_ |
| `GOOGLE_API_KEY` | _(unset)_ | Google AI API key |
| `GOOGLE_API_MODEL` | _(unset)_ | e.g. `gemini-2.5-flash` |

The `.env` file carries cloud credentials. The `.env-local` file holds the Docker Compose service hostnames (commented out by default — uncomment to override for local container networking).

### Running Locally Without a GPU

The `docker-compose.yml` configures Ollama with NVIDIA GPU access. To run on CPU only, remove the `deploy` block from the `ollama` service:

```yaml
# Remove this block for CPU-only mode:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Then switch to a smaller model in `config.py`:

```python
OLLAMA_MODEL = "llama3.2:1b"   # fast on CPU
OLLAMA_MODEL = "phi3:mini"      # good balance
OLLAMA_MODEL = "gemma3:1b"      # compact Gemma variant
```

And update the model pulled in `docker-compose.yml`:

```yaml
ollama-init:
  command: ["pull", "llama3.2:1b"]   # match config.py
```

> The fastembed embedding models (BGE + CLIP) run entirely on CPU and need no GPU in either mode. Only Ollama benefits from GPU acceleration.

### Switching to Cloud Mode

Set these variables in `.env` and leave `OLLAMA_HOST` / `OLLAMA_MODEL` in `.env-local`:

```bash
GOOGLE_API_KEY=<your-google-ai-key>
GOOGLE_API_MODEL=gemini-2.5-flash
QDRANT_HOST=<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-key>
REDIS_URL=redis://<your-redis-host>:<port>
REDIS_PASSWORD=<your-redis-password>
```

In cloud mode the `ollama` and `ollama-init` services in `docker-compose.yml` are unused (the app container will start without them if you remove their `depends_on` entry), but they will still start unless you remove them from the compose file.

---

## Startup Flow

### Local (docker-compose up)

```
docker-compose up
       │
       ├─► Redis Stack starts    (port 6379)
       ├─► Qdrant starts         (port 6333)
       └─► Ollama starts         (port 11434)
                │
                ▼
         ollama-init waits for Ollama health check
                │
                ▼
         ollama pull gemma4:e4b  (downloads model weights — once)
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

### Cloud (run app directly)

```
Set env vars (GOOGLE_API_KEY, QDRANT_API_KEY, REDIS_URL, ...)
       │
       ▼
python data_loader/pokemon_loader.py
       │   Fetches ~1,500 Pokémon, embeds, indexes into Qdrant Cloud
       │   (skipped if collections already exist)
       │
       ▼
streamlit run streamlit_app/app.py
       │
       ▼
http://localhost:8501
```

On first run the ingestion step takes several minutes (network + embedding). On subsequent runs it is skipped and the app starts in seconds.
