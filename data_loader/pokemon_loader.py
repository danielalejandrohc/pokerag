import base64
import io
import os
import time
import threading
import concurrent.futures

import requests
from PIL import Image
from fastembed import ImageEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

COLLECTION_NAME = "pokemons"
IMAGES_COLLECTION_NAME = "pokemon_images"
TEXT_VECTOR_SIZE = 384   # BAAI/bge-small-en-v1.5
IMAGE_VECTOR_SIZE = 512  # Qdrant/clip-ViT-B-32-vision
MAX_WORKERS = 10

_failed_pokemon_ids = set()
_text_model: TextEmbedding | None = None
_image_model: ImageEmbedding | None = None
_text_model_lock = threading.Lock()
_image_model_lock = threading.Lock()


def _get_text_model() -> TextEmbedding:
    global _text_model
    with _text_model_lock:
        if _text_model is None:
            _text_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _text_model


def _get_image_model() -> ImageEmbedding:
    global _image_model
    with _image_model_lock:
        if _image_model is None:
            _image_model = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")
    return _image_model


def _pokemon_to_text(data: dict) -> str:
    types = ", ".join(t["type"]["name"] for t in data.get("types", []))
    abilities = ", ".join(a["ability"]["name"] for a in data.get("abilities", []))
    stats = ", ".join(
        f"{s['stat']['name']}: {s['base_stat']}" for s in data.get("stats", [])
    )
    moves = ", ".join(m["move"]["name"] for m in data.get("moves", []))
    return (
        f"Name: {data['name']}. "
        f"Types: {types}. "
        f"Abilities: {abilities}. "
        f"Stats: {stats}. "
        f"Moves: {moves}. "
        f"Height: {data.get('height')}. Weight: {data.get('weight')}."
    )


def get_qdrant_client() -> QdrantClient:
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=api_key)
    else:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collections_exist(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=TEXT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="name",
            field_schema="keyword",
        )
        print(f"Created collection '{COLLECTION_NAME}'")

    if IMAGES_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=IMAGES_COLLECTION_NAME,
            vectors_config=VectorParams(size=IMAGE_VECTOR_SIZE, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=IMAGES_COLLECTION_NAME,
            field_name="name",
            field_schema="keyword",
        )
        print(f"Created collection '{IMAGES_COLLECTION_NAME}'")


def get_pokemon_data_in_local_memory() -> list[dict]:
    url = "https://pokeapi.co/api/v2/pokemon?limit=1500"
    response = requests.get(url)
    return response.json()["results"]


def pull_pokemon_data_with_image(pokemon: dict) -> dict:
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon['name']}"
    pokemon_data = requests.get(url).json()

    image_bytes = None
    artwork_url = (
        pokemon_data["sprites"]
        .get("other", {})
        .get("official-artwork", {})
        .get("front_default")
    )
    if artwork_url:
        img_response = requests.get(artwork_url)
        if img_response.ok:
            image_bytes = img_response.content

    return {"data": pokemon_data, "image": image_bytes, "artwork_url": artwork_url}


def _store_pokemon(client: QdrantClient, entry: dict) -> None:
    pokemon_data = entry["data"]
    image_bytes = entry["image"]
    artwork_url = entry["artwork_url"]
    pokemon_id = pokemon_data["id"]

    text_vector = next(iter(_get_text_model().embed([_pokemon_to_text(pokemon_data)]))).tolist()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=pokemon_id,
                vector=text_vector,
                payload={
                    "name": pokemon_data["name"],
                    "id": pokemon_id,
                    "types": [t["type"]["name"] for t in pokemon_data.get("types", [])],
                    "abilities": [a["ability"]["name"] for a in pokemon_data.get("abilities", [])],
                    "stats": {s["stat"]["name"]: s["base_stat"] for s in pokemon_data.get("stats", [])},
                    "height": pokemon_data.get("height"),
                    "weight": pokemon_data.get("weight"),
                    "artwork_url": artwork_url,
                    "data": pokemon_data,
                },
            )
        ],
    )

    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes))
        clip_vector = next(iter(_get_image_model().embed([image]))).tolist()

        client.upsert(
            collection_name=IMAGES_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=pokemon_id,
                    vector=clip_vector,
                    payload={
                        "pokemon_id": pokemon_id,
                        "name": pokemon_data["name"],
                        "artwork_url": artwork_url,
                        "image_b64": base64.b64encode(image_bytes).decode(),
                    },
                )
            ],
        )


def wait_for_qdrant(max_retries: int = 30, delay: int = 2) -> QdrantClient:
    for attempt in range(1, max_retries + 1):
        try:
            client = get_qdrant_client()
            client.get_collections()
            print("Qdrant is ready.")
            return client
        except Exception as exc:
            print(f"Waiting for Qdrant ({attempt}/{max_retries}): {exc}")
            time.sleep(delay)
    raise RuntimeError("Qdrant did not become ready in time.")


def collections_are_ready(client: QdrantClient) -> bool:
    existing = {c.name for c in client.get_collections().collections}
    return COLLECTION_NAME in existing and IMAGES_COLLECTION_NAME in existing


def process_prokemon_data(pokemon_list: list[dict]) -> None:
    client = get_qdrant_client()
    ensure_collections_exist(client)
    global _failed_pokemon_ids
    _failed_pokemon_ids = set()

    def fetch_and_store(pokemon: dict) -> None:
        try:
            entry = pull_pokemon_data_with_image(pokemon)
            _store_pokemon(client, entry)
            print(f"[OK] {pokemon['name']}")
        except Exception as exc:
            print(f"[FAIL] {pokemon['name']}: {exc}")
            _failed_pokemon_ids.add(pokemon["id"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(fetch_and_store, pokemon_list)


if __name__ == "__main__":
    client = wait_for_qdrant()
    if collections_are_ready(client):
        print("Both collections already exist — skipping load.")
    else:
        print("One or more collections missing — starting full load.")
        data = get_pokemon_data_in_local_memory()
        process_prokemon_data(data)
        if _failed_pokemon_ids:
            time.sleep(5)  # Brief pause before retrying failed entries
            print(f"Failed to load data for Pokémon IDs: {_failed_pokemon_ids}")
            retry_list = [p for p in data if p["id"] in _failed_pokemon_ids]
            process_prokemon_data(retry_list)