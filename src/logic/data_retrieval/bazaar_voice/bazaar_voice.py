"""
pip install git+https://github.com/danieldanieltata/Bazaarvoice-API.git
pip install tenacity
pip install pydantic-settings
"""
import argparse
import requests
import logging
from enum import Enum
from tqdm import tqdm
from typing import List, Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from bazaarvoice_api import BazaarvoiceAPI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.logic.data_retrieval.bazaar_voice.config import BazaarvoiceSettings

settings = BazaarvoiceSettings()


class ReviewKeys(str, Enum):
    """Enumeration of standard review keys returned by Bazaarvoice API."""
    USER_LOCATION = "UserLocation"
    CONTENT_LOCALE = "ContentLocale"
    RATING = "Rating"
    SUBMISSION_TIME = "SubmissionTime"
    REVIEW_TEXT = "ReviewText"
    TITLE = "Title"
    IS_RECOMMENDED = "IsRecommended"
    PRODUCT_ID = "ProductId"
    EAN = "EAN"


class ContextAttributes(str, Enum):
    """Enumeration of context attribute keys returned by Bazaarvoice API."""
    AGE = "Age"
    BEAUTY_EXPERTISE = "BeautyExpertise"
    EYE_COLOR = "EyeColor"
    REPURCHASE = "Wouldyoupurchaseagain"
    SKIN_TONE = "SkinTone"
    QUALITY = "Quality"


class TagDimensions(str, Enum):
    """Enumeration of tag dimensions keys returned by Bazaarvoice API."""
    PRODUCT_USAGE = "ProductUsage"


def setup_logging(verbose: bool = False):
    """
    Configures logging for the application.

    Args:
        verbose: Whether to enable debug-level logging.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # Silence noisy loggers from third-party libraries (optional)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


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
        f"?apiversion=5.4&passkey={token}"
        f"&Filter=ProductId:{product_id}&offset={offset}&limit={limit}"
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
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
    headers = settings.DEFAULT_HEADERS
    url = generate_product_reviews_url(product_id, offset, limit=limit)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"[{product_id}] Request failed at offset {offset}: {e}")
        raise

    try:
        return response.json().get("Results", [])
    except ValueError as e:
        logging.error(f"[{product_id}] Failed to parse JSON at offset {offset}: {e}")
        raise


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def get_total_reviews(product_id: str, limit: int = 1) -> int:
    """
    Retrieves the total number of available reviews for a product.

    Args:
        product_id: Product ID.
        limit: Number of reviews to request (only the total count is used).

    Returns:
        The total number of reviews available.
    """
    headers = settings.DEFAULT_HEADERS
    url = generate_product_reviews_url(product_id, 0, limit=limit)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"[{product_id}] Error fetching total reviews: {e}")
        raise

    try:
        return response.json().get("TotalResults", 0)
    except ValueError as e:
        logging.error(f"[{product_id}] Invalid JSON when retrieving total reviews: {e}")
        raise


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

    # First, get total number of reviews (for batching)
    try:
        total_available = get_total_reviews(product_id, limit=1)
    except requests.RequestException as e:
        logging.error(f"[{product_id}] Failed to get total reviews after retries: {e}")
        return []

    if total_available == 0:
        return []

    # Modified Version (Memory Efficient):
    offsets = range(0, total_available, limit)
    reviews = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_offset = {
            executor.submit(fetch_review_batch, product_id, offset, limit): offset
            for offset in offsets
        }

        for future in tqdm(as_completed(future_to_offset), total=total_available // limit,
                           desc=f"Fetching **product_id = {product_id}** reviews"):
            try:
                result = future.result()
                reviews.extend(result)
            except Exception as e:
                logging.error(f"Error fetching batch at offset {future_to_offset[future]}: {e}")

    return reviews


def process_review(review: dict[str, Any], prod: Any) -> Optional[dict[str, Any]]:
    """
    Processes a single review and extracts relevant fields and attributes.

    Args:
        review: The raw review dictionary.
        prod: The product object associated with the review.

    Returns:
        A dictionary mapping review ID to its structured data, or None if invalid.
    """
    product_id = review.get('ProductId')
    if product_id != prod.Id:
        return None

    id_ = review.get('Id')
    if not id_:
        return None

    data = {
        ReviewKeys.USER_LOCATION: review.get(ReviewKeys.USER_LOCATION),
        ReviewKeys.CONTENT_LOCALE: review.get(ReviewKeys.CONTENT_LOCALE),
        ReviewKeys.RATING: review.get(ReviewKeys.RATING),
        ReviewKeys.SUBMISSION_TIME: review.get(ReviewKeys.SUBMISSION_TIME),
        ReviewKeys.REVIEW_TEXT: review.get(ReviewKeys.REVIEW_TEXT),
        ReviewKeys.TITLE: review.get(ReviewKeys.TITLE),
        ReviewKeys.IS_RECOMMENDED: review.get(ReviewKeys.IS_RECOMMENDED),
        ReviewKeys.PRODUCT_ID: product_id,
        ReviewKeys.EAN: prod.EANs[0] if prod.EANs else None,
    }

    context = review.get('ContextDataValues')
    if context:
        data.update({
            ContextAttributes.AGE: get_attribute(context, ContextAttributes.AGE),
            ContextAttributes.BEAUTY_EXPERTISE: get_attribute(context, ContextAttributes.BEAUTY_EXPERTISE),
            ContextAttributes.EYE_COLOR: get_attribute(context, ContextAttributes.EYE_COLOR),
            ContextAttributes.REPURCHASE: get_attribute(context, ContextAttributes.REPURCHASE),
            ContextAttributes.SKIN_TONE: get_attribute(context, ContextAttributes.SKIN_TONE),
            ContextAttributes.QUALITY: get_attribute(context, ContextAttributes.QUALITY),
        })

        tagdimensions = context.get('TagDimensions')
        if tagdimensions:
            data[TagDimensions.PRODUCT_USAGE] = get_attribute(tagdimensions, TagDimensions.PRODUCT_USAGE)

    return {id_: data}


def handle_product_reviews(prod: Any, max_workers: int, limit: int) -> dict[str, dict[str, Any]]:
    """
    Retrieves and processes all reviews for a single product.

    Args:
        prod: The product object.
        max_workers: Number of threads for concurrent execution.
        limit: Number of reviews per request batch.

    Returns:
        A dictionary of processed reviews keyed by review ID.
    """
    product_id = prod.Id
    reviews = generate_reviews_by_id(product_id=product_id, max_workers=max_workers, limit=limit)

    # Use dictionary comprehension instead of update in loop
    processed_reviews = {
        rid: pdata
        for review in reviews
        # https://stackoverflow.com/questions/26000198/what-does-colon-equal-in-python-mean
        if (processed := process_review(review, prod)) is not None
        for rid, pdata in processed.items()
    }

    return processed_reviews


def main(brand: str, max_workers: int, limit: int) -> defaultdict:
    """
    Entry point for fetching all reviews from the Bazaarvoice API for a given brand.

    Args:
        brand: Brand name to fetch products for.
        max_workers: Number of threads to use.
        limit: Number of reviews per request.

    Returns:
        A defaultdict mapping review IDs to structured review data.
    """
    bazzarvoice = BazaarvoiceAPI(settings.BAZAAR_VOICE_TOKEN, brand)
    data = defaultdict(dict)

    for prod in bazzarvoice.get_product():
        product_reviews = handle_product_reviews(prod, max_workers=max_workers, limit=limit)
        data.update(product_reviews)

    return data


def get_attribute(context: dict[str, Any], attribute: str) -> Optional[str]:
    """
    Extracts an attribute's value from the context data.

    Args:
        context: Dictionary containing context or tag dimension data.
        attribute: Key of the attribute to retrieve.

    Returns:
        The value of the attribute if found, else None.
    """

    att = context.get(attribute)

    if att is None:
        return att
    else:
        return att['Value']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fetch Bazaarvoice product reviews.")
    parser.add_argument('--brand', help="Brand name to query.", required=True)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of concurrent threads to use for fetching reviews (default: 8)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of reviews per request batch (default: 100)"
    )
    args = parser.parse_args()

    main(brand=args.brand, max_workers=args.max_workers, limit=args.limit)
