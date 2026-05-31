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

# ── Qdrant collection ──────────────────────────────────────────────────────────
# Single collection with named vectors: "text" (BGE) and "image" (CLIP).
# Each point holds the full Pokemon payload including image_b64.
COLLECTION_NAME = "pokemons"

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
TEXT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CACHE_EMBED_MODEL = "redis/langcache-embed-v1"
IMAGE_EMBED_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CLIP_TEXT_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# ── Search ─────────────────────────────────────────────────────────────────────
MAX_RESULTS = 30   # max Qdrant hits returned per search query
N_COLS = 5         # number of columns in the results grid
SCORE_THRESHOLD = 0.55        # BGE text-to-text cosine similarity (tight range: 0.6–0.9)
IMAGE_SCORE_THRESHOLD = 0.20  # CLIP text/image-to-image cosine similarity (loose range: 0.2–0.35)

# Only these payload fields are fetched from Qdrant to keep responses lean.
# The full "data" field (raw PokeAPI JSON) is intentionally excluded here;
# it is only fetched when the RAG context or tool needs the complete moves list.
TEXT_PAYLOAD = ["name", "id", "types", "stats", "height", "weight", "artwork_url", "image_b64"]
IMAGE_PAYLOAD = ["id", "name", "artwork_url", "image_b64"]

# ── Semantic cache ─────────────────────────────────────────────────────────────
CACHE_TTL = 3600 * 48   # entries expire after 2 days
CACHE_DISTANCE_THRESHOLD = 0.135  # cosine distance threshold (= 1 − 0.80 similarity)
                                 # 0.135 was too tight for short conversational questions —
                                 # "how it look?" and "how does it look?" were both missing
