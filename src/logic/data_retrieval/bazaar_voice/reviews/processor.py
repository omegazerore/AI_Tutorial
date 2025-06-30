from typing import Any, Optional, List
from .constants import ReviewKeys, ContextAttributes, TagDimensions
from .fetcher import generate_reviews_by_id


def get_attribute(context: dict[str, Any], attribute: str) -> Optional[str]:
    """
    Extracts an attribute's value from the context data.

    Args:
        context: Dictionary containing context or tag dimension data.
        attribute: Key of the attribute to retrieve.

    Returns:
        The value of the attribute if found, else None.
    """
    attr = context.get(attribute)
    return attr['Value'] if attr and 'Value' in attr else None


def process_review(review: dict[str, Any], prod: Any) -> Optional[dict[str, Any]]:
    """
    Processes a single review and extracts relevant fields and attributes.

    Args:
        review: The raw review dictionary.
        prod: The product object associated with the review.

    Returns:
        A dictionary mapping review ID to its structured data, or None if invalid.
    """
    product_id = review.get("ProductId")
    if product_id != prod.Id or not review.get("Id"):
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

    context = review.get("ContextDataValues", {})
    data.update({
        ContextAttributes.AGE: get_attribute(context, ContextAttributes.AGE),
        ContextAttributes.BEAUTY_EXPERTISE: get_attribute(context, ContextAttributes.BEAUTY_EXPERTISE),
        ContextAttributes.EYE_COLOR: get_attribute(context, ContextAttributes.EYE_COLOR),
        ContextAttributes.REPURCHASE: get_attribute(context, ContextAttributes.REPURCHASE),
        ContextAttributes.SKIN_TONE: get_attribute(context, ContextAttributes.SKIN_TONE),
        ContextAttributes.QUALITY: get_attribute(context, ContextAttributes.QUALITY),
    })

    tagdimensions = context.get("TagDimensions")
    if tagdimensions:
        data[TagDimensions.PRODUCT_USAGE] = get_attribute(tagdimensions, TagDimensions.PRODUCT_USAGE)

    return {review["Id"]: data}


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
    reviews = generate_reviews_by_id(prod.Id, limit, max_workers)
    return {
        rid: pdata
        for review in reviews
        if (processed := process_review(review, prod)) is not None
        for rid, pdata in processed.items()
    }