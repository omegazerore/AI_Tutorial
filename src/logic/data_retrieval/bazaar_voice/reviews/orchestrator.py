from collections import defaultdict
from src.logic.data_retrieval.bazaar_voice.config import BazaarvoiceSettings
from bazaarvoice_api import BazaarvoiceAPI
from .processor import handle_product_reviews

settings = BazaarvoiceSettings()

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
        reviews = handle_product_reviews(prod, max_workers, limit)
        data.update(reviews)

    return data
