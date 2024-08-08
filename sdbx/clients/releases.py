import os
import aiofiles
import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import aiohttp
import asyncio

SERVICES_TO_CHECK = {
    "github": lambda user, project, asset: f"https://github.com/{user}/{project}/releases/latest/download/{asset}",
    "gitlab": lambda namespace, project, path: f"https://gitlab.com/{namespace}/{project}/-/releases/permalink/latest/downloads:{path}",
    # TODO: add more (bitbucket, sourceforge?)
}

ASSETS_TO_CHECK = ["dist.zip", "build.zip"]

def parse_service(url, signature):
    service = next((k for k in SERVICES_TO_CHECK.keys() if k in url), None)
    namespace, project = signature.rsplit("/", 1)
    return namespace, project, service

async def get_asset_url(session, namespace, project, service=None):
    services = { service: SERVICES_TO_CHECK[service] } if service else SERVICES_TO_CHECK

    for s in services.values():
        for asset in ASSETS_TO_CHECK:
            url = s(namespace, project, asset)
            try:
                async with session.head(url, allow_redirects=True) as response:
                    if response.status == 404:
                        logging.debug(f"404 Not Found: {url}")
                        continue
                    
                    lastmodified = response.headers.get('last-modified', None)

                    if not lastmodified:
                        logging.error(f"No last-modified header found for {url}")

                    return response.url, parsedate_to_datetime(lastmodified)
            except Exception as e:
                logging.debug(f"Error fetching {url}: {e}")
                continue
    
    return None

async def download_asset(session, url, extract_path):    
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                
               # Use a temporary file to save the zip content
                with tempfile.NamedTemporaryFile() as zip:
                    async with aiofiles.open(zip.name, 'wb') as f:
                        await f.write(content)

                    # Wipe the directory first
                    if os.path.exists(extract_path):
                        shutil.rmtree(extract_path)
                    os.makedirs(extract_path, exist_ok=True)
                
                    # Extract the zip file into the directory
                    with zipfile.ZipFile(zip.name, 'r') as ref:
                        ref.extractall(extract_path)
                    
                    # Search for index.html recursively within the client directory
                    for root, dirs, files in os.walk(extract_path):
                        if "index.html" in files:
                            with tempfile.TemporaryDirectory() as copydir:
                                temproot = shutil.copytree(root, os.path.join(copydir, os.path.basename(root)))
                                shutil.rmtree(extract_path)
                                shutil.move(temproot, extract_path) # Overwrite old directory with true root
                
                logging.debug(f"Downloaded and extracted asset from {url} to {extract_path}")
            else:
                raise Exception(f"Failed to download asset from {url}, status code: {response.status}")
    except Exception as e:
        raise Exception(f"Error downloading {url}: {e}")