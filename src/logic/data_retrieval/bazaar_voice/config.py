"""
Configuration module for Bazaarvoice API integration.

Loads credentials from an INI file and sets up the configuration using pydantic-based settings.
"""
import os
import configparser
from pydantic import Field
from pydantic_settings import BaseSettings

from src.io.path_definition import get_file

# Path to the credentials file
credential_file = get_file("config/credentials.ini")

# Load Bazaarvoice token into environment if file exists
if os.path.exists(credential_file):
    config_ = configparser.ConfigParser()
    config_.read(credential_file)
    os.environ['BAZAAR_VOICE'] = config_['bazaar_voice'].get('token')


class BazaarvoiceSettings(BaseSettings):
    """
    Configuration settings for Bazaarvoice API access.
    """

    BAZAAR_VOICE_TOKEN: str = Field(os.environ['BAZAAR_VOICE'])
    BAZAAR_VOICE_BASE_URL: str = "https://api.bazaarvoice.com"
    DEFAULT_HEADERS: dict = {"User-Agent": (
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'
    )}