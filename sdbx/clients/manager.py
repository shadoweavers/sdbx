import os
import tomllib
import logging

import aiohttp

from sdbx.clients.releases import download_asset, get_asset_url, parse_service
from sdbx.utils import aobject

class ClientManager(aobject):
    async def __init__(self, path, clients_path):
        self.path = path
        self.clients_path = clients_path
        
        with open(os.path.join(path, "extensions.toml"), 'rb') as file:
            self.client_signatures = tomllib.load(file)["clients"]
        
        if not self.client_signatures:
            # trigger remote/embedded client or something lol
            return

        self.selected_path = await self.validate_clients_installed()
    
    async def validate_clients_installed(self):
        first_viable = None

        async with aiohttp.ClientSession() as session:
            for client_signature, url in self.client_signatures.items():
                client_path = os.path.join(self.clients_path, os.path.normpath(client_signature))

                if not os.path.exists(client_path):
                    # os.makedirs(client_path, exist_ok=True)
                    logging.info(f"Client {client_signature} not installed, downloading...")
                    namespace, project, service = parse_service(url, client_signature)
                    asset_url, _ = await get_asset_url(session, namespace, project, service=service)
                    await download_asset(session, asset_url, client_path)

                if os.path.exists(os.path.join(client_path, "index.html")) and first_viable is None:
                    first_viable = client_path
            
        if first_viable == None:
            raise Exception("No viable clients could be found. Check your installations.")

        return first_viable
    
    async def update_clients(self):
        for client_signature, url in self.client_signatures.items():
            client_path = os.path.join(self.clients_path, os.path.normpath(client_signature))

            if not os.path.exists(client_path):
                continue # skip downloading new clients, that's the job of startup

            namespace, project, service = parse_service(url, client_signature)                
            asset_url, lastmodified = get_asset_url(namespace, project, service=service)
            if os.path.getmtime(client_path) < lastmodified:
                download_asset(asset_url, client_path)