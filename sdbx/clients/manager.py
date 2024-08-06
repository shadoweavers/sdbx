import os
import sys
import site
import venv
import tomllib
import subprocess

from dulwich import porcelain

from sdbx.clients.releases import download_asset, get_asset_url, parse_service

class ClientManager:
    def __init__(self, path, clients_path):
        self.path = path
        self.clients_path = clients_path
        
        with open(os.path.join(path, "extensions.toml"), 'rb') as file:
            self.client_signatures = tomllib.load(file)["clients"]
        
        if not self.client_signatures:
            # trigger remote/embedded client or something lol
            return

        self.validate_clients_installed()

        self.selected_path = os.path.join(self.clients_path, os.path.normpath(self.selected))
    
    def validate_clients_installed(self):
        for client_signature, url in self.client_signatures.items():
            client_path = os.path.join(self.clients_path, os.path.normpath(client_signature))

            if not os.path.exists(client_path):
                os.makedirs(client_path, exist_ok=True)
                namespace, project, service = parse_service(url, client_signature)
                asset_url, _ = get_asset_url(namespace, project, service=service)
                download_asset(asset_url, client_path)
    
    def update_clients(self):
        for client_signature, url in self.client_signatures.items():
            client_path = os.path.join(self.clients_path, os.path.normpath(client_signature))

            if not os.path.exists(client_path):
                continue # skip downloading new clients, that's the job of startup

            namespace, project, service = parse_service(url, client_signature)                
            asset_url, lastmodified = get_asset_url(namespace, project, service=service)
            if os.path.getmtime(client_path) < lastmodified:
                download_asset(asset_url, client_path)