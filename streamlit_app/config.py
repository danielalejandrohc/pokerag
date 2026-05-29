import os

# ── Infrastructure ─────────────────────────────────────────────────────────────
# All values are read from environment variables so the same image works in
# Docker Compose (where services talk by container name) and locally
# (where everything runs on localhost).

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")  # change to any Ollama model name

# ── Qdrant collections ─────────────────────────────────────────────────────────
# Two separate collections: one stores text embeddings (stats, moves, etc.),
# the other stores CLIP image embeddings (visual similarity).
COLLECTION_TEXT = "pokemons"
COLLECTION_IMAGES = "pokemon_images"

# ── Embedding models ───────────────────────────────────────────────────────────
# TEXT_EMBED_MODEL  — BGE model used for Qdrant text search. Must match the
#                     model used when the collection was originally indexed.
# CACHE_EMBED_MODEL — Separate model used only by the semantic cache. Redis's
#                     langcache-embed-v1 is fine-tuned for question paraphrase
#                     detection, making it a better fit than a general-purpose
#                     search model for deciding "is this the same question?".
# IMAGE_EMBED_MODEL — CLIP vision encoder; embeds actual Pokemon artwork.
# CLIP_TEXT_MODEL   — CLIP text encoder; same vector space as the vision
#                     encoder, so text phrases can be compared against images.
TEXT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CACHE_EMBED_MODEL = "redis/langcache-embed-v1"
IMAGE_EMBED_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"

# ── Search ─────────────────────────────────────────────────────────────────────
MAX_RESULTS = 10   # max Qdrant hits returned per search query
N_COLS = 5         # number of columns in the results grid
SCORE_THRESHOLD = 0.55  # minimum similarity score for a hit to be included in results

# Only these payload fields are fetched from Qdrant to keep responses lean.
# The full "data" field (raw PokeAPI JSON) is intentionally excluded here;
# it is only fetched when the RAG context or tool needs the complete moves list.
TEXT_PAYLOAD = ["name", "id", "types", "stats", "height", "weight", "artwork_url"]
IMAGE_PAYLOAD = ["pokemon_id", "name", "artwork_url", "image_b64"]

# ── Semantic cache ─────────────────────────────────────────────────────────────
CACHE_TTL = 3600 * 48   # entries expire after 2 days
CACHE_DISTANCE_THRESHOLD = 0.135  # cosine distance threshold (= 1 − 0.80 similarity)
                                 # 0.135 was too tight for short conversational questions —
                                 # "how it look?" and "how does it look?" were both missing
