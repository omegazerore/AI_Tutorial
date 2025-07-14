"""
Downloads product sitemap XML files sequentially from icheck.vn and saves them locally.
Retrieves files by incrementing page IDs until a missing file is detected or a retry limit is reached.
"""
import logging
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.io.path_definition import get_datafetch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
URL_TEMPLATE = "https://icheck.vn/sitemap/products/products-{page_id}.xml"
SAVE_PATH_TEMPLATE = "products-{page_id}.xml"
RETRY_LIMIT = 100
BATCH_SIZE = 10


def get_last_page_id(save_dir: str) -> int:
    """Get the highest page_id from existing files in save_dir."""
    pattern = re.compile(r'^products-(\d+)\.xml$')
    max_id = -1
    if not os.path.exists(save_dir):
        return -1
    for filename in os.listdir(save_dir):
        match = pattern.match(filename)
        if match:
            page_id = int(match.group(1))
            if page_id > max_id:
                max_id = page_id
    return max_id


def download_file(url: str, save_path: str):
    """Download a file from a URL and save it to a local path.

    Sends an HTTP GET request to the specified URL and writes the response content to the given local file path.
    Raises an exception for HTTP errors (including 4xx and 5xx status codes).

    Args:
        url: The URL of the file to be downloaded.
        save_path: The local filesystem path where the file will be saved.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails or returns an error status code.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 404:
            logger.info(f"File not found (404): {url}")
            return False
        response.raise_for_status()  # raise an exception for 400/500 response
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        logger.info(f"File downloaded successfully and saved to {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def main() -> None:
    """Download product sitemap XML files sequentially.

    Increments the page ID to construct URLs and downloads each XML file, saving them under a local directory.
    Stops when a file download fails consecutively for RETRY_LIMIT attempts (such as repeated 404 errors).

    Returns:
        None
    """
    save_dir = os.path.join(get_datafetch(), "icheck")
    os.makedirs(save_dir, exist_ok=True)

    last_page_id = get_last_page_id(save_dir)
    start_page_id = last_page_id + 1 if last_page_id >= 0 else 0
    logger.info(f"Starting from page_id {start_page_id}")

    page_id = start_page_id
    consecutive_failures = 0

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        while consecutive_failures < RETRY_LIMIT:
            batch = []
            for _ in range(BATCH_SIZE):
                save_path = os.path.join(save_dir, SAVE_PATH_TEMPLATE.format(page_id=page_id))
                url = URL_TEMPLATE.format(page_id=page_id)
                if not os.path.isfile(save_path):
                    batch.append((page_id, url, save_path))
                page_id += 1

            if len(batch) == 0:
                continue

            # Submit tasks for batch
            future_to_page = {
                executor.submit(download_file, url, save_path): page_id
                for page_id, url, save_path in batch
            }

            # Check results
            for future in as_completed(future_to_page):
                page_id_done = future_to_page[future]
                try:
                    success = future.result()
                    if success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except Exception as e:
                    logger.warning(f"Error for page {page_id_done}: {e}")
                    consecutive_failures += 1

                if consecutive_failures >= RETRY_LIMIT:
                    logger.info(f"Stopping: {RETRY_LIMIT} consecutive failures.")
                    break

    logger.info("Download process completed.")

if __name__ == "__main__":

    main()