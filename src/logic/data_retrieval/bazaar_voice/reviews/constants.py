from enum import Enum

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
