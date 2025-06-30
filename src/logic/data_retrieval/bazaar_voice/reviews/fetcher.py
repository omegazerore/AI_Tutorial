import logging
import requests
from typing import Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.logic.data_retrieval.bazaar_voice.config import BazaarvoiceSettings

settings = BazaarvoiceSettings()

def generate_product_reviews_url(product_id: str, offset: int, limit: int) -> str:
    """
    Constructs the API URL to fetch product reviews.

    Args:
        product_id: ID of the product to fetch reviews for.
        offset: Pagination offset.
        limit: Number of reviews to fetch in a single request.

    Returns:
        A formatted URL string.
    """
    token = settings.BAZAAR_VOICE_TOKEN
    return (
        f"{settings.BAZAAR_VOICE_BASE_URL}/data/reviews.json"
        f"?apiversion=5.4&passkey={token}&Filter=ProductId:{product_id}&offset={offset}&limit={limit}"
    )


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(requests.RequestException))
def fetch_review_batch(product_id: str, offset: int, limit: int = 100) -> List[dict[str, Any]]:
    """
    Fetches a batch of product reviews using Bazaarvoice API.

    Args:
        product_id: ID of the product to fetch reviews for.
        offset: Offset for pagination.
        limit: Number of reviews to retrieve in this batch.

    Returns:
        A list of review dictionaries.
    """
    url = generate_product_reviews_url(product_id, offset, limit)
    response = requests.get(url, headers=settings.DEFAULT_HEADERS, timeout=10)
    response.raise_for_status()
    return response.json().get("Results", [])


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(requests.RequestException))
def get_total_reviews(product_id: str, limit: int = 1) -> int:
    """
    Retrieves the total number of available reviews for a product.

    Args:
        product_id: Product ID.
        limit: Number of reviews to request (only the total count is used).

    Returns:
        The total number of reviews available.
    """
    url = generate_product_reviews_url(product_id, 0, limit)
    response = requests.get(url, headers=settings.DEFAULT_HEADERS, timeout=10)
    response.raise_for_status()
    return response.json().get("TotalResults", 0)


def generate_reviews_by_id(product_id: str, limit: int, max_workers: int) -> List[dict[str, Any]]:
    """
    Concurrently fetches all reviews for a given product using threads.

    Args:
        product_id: Product ID.
        limit: Number of reviews per API request.
        max_workers: Number of threads to use for concurrent fetching.

    Returns:
        A list of review dictionaries.
    """
    total = get_total_reviews(product_id)
    if total == 0:
        return []

    offsets = range(0, total, limit)
    reviews = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_offset = {
            executor.submit(fetch_review_batch, product_id, offset, limit): offset
            for offset in offsets
        }

        for future in tqdm(as_completed(future_to_offset), total=len(offsets), desc=f"Fetching reviews for {product_id}"):
            try:
                reviews.extend(future.result())
            except Exception as e:
                logging.error(f"Error fetching offset {future_to_offset[future]}: {e}")

    return reviews
