import logging

from redisvl.query.filter import Tag

from clients import get_semantic_cache
from config import CACHE_DISTANCE_THRESHOLD

logger = logging.getLogger(__name__)

# Fetch the nearest entry regardless of threshold so we can always log its distance.
# The actual hit/miss decision is made in Python below.
_BROAD_THRESHOLD = 1.0


def cache_lookup(pokemon_name: str, question: str) -> str | None:
    """Return a cached answer if a semantically similar question exists for this Pokemon."""
    try:
        logger.info("Cache lookup [%s] q=%r", pokemon_name, question)
        cache = get_semantic_cache()
        results = cache.check(
            prompt=f"{pokemon_name} {question}",
            num_results=5,
        )
        # debug results
        for r in results:
            logger.debug(
                "Cache candidate [%s][%s] distance=%.4f  entire_row=%s",
                pokemon_name,
                question,
                r.get("vector_distance", float("inf")),
                str(r),
            )

        if results:
            distance = results[0].get("vector_distance", float("inf"))
            logger.debug(
                "Cache HIT [%s] nearest=%.4f (threshold=%.2f)  q=%r",
                pokemon_name, distance, CACHE_DISTANCE_THRESHOLD, question,
            )
            return results[0]["response"]
        else:
            logger.debug("Cache MISS [%s] (no entries yet)  q=%r", pokemon_name, question)
    except Exception as exc:
        logger.warning("Cache lookup failed [%s]: %s", pokemon_name, exc, exc_info=True)

    return None


def cache_store(pokemon_name: str, question: str, answer: str) -> None:
    """Store a question/answer pair for this Pokemon in Redis."""
    try:
        get_semantic_cache().store(
            prompt=f"{pokemon_name} {question}",
            response=answer,
        )
        logger.debug("Cache STORE [%s] q=%r", pokemon_name, question)
    except Exception as exc:
        logger.warning("Cache store failed [%s]: %s", pokemon_name, exc, exc_info=True)
