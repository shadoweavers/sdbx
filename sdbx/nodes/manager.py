import os
import sys
import site
import venv
import logging
import tomllib
import subprocess
from importlib.metadata import distribution

from dulwich import porcelain

from sdbx.nodes.adapters import ComfyAdapter
from sdbx.nodes.base import register_base_nodes

PYPY_URL = "https://github.com/pypy/pypy.git"

class NodeManager:
    registry = {}

    def __init__(self, path, nodes_path, env_name=".node_env"):
        self.path = path
        self.nodes_path = nodes_path
        
        with open(os.path.join(path, "extensions.toml"), 'rb') as file:
            self.node_modules = tomllib.load(file)["nodes"]
        
        self.initialize_environment(env_name)

        self.nodes = [self.validate_nodes_installed(n, u) for n, u in self.node_modules.items()]
    
    def initialize_environment(self, env_name=".node_env"):
        logging.info("Creating node environment...")

        self.env_path = os.path.join(self.path, env_name)

        if not os.path.exists(self.env_path):
            venv.create(self.env_path, with_pip=True)
        
        # Grab interpreter + sandbox
        pypy_path = os.path.join(self.env_path, "pypy")
        porcelain.clone(PYPY_URL, pypy_path)
        
        self.env_python = os.path.join(self.env_path, "bin", "python")
        self.env_package_path = self.get_environment_package_path()

        sys.path.append(self.env_package_path) # Add node env packages to our current environemnt

        parent_package_path = site.getsitepackages()[0]
        parent_link_path = os.path.join(self.env_package_path, "parent.pth")
        if not os.path.exists(parent_link_path):
            with open(parent_link_path, 'w') as f:
                f.write(parent_package_path + "\n")
                f.write(pypy_path)

        self.env_pip = os.path.join(self.env_path, "bin", "pip")
        self.env_packages = self.get_environment_packages()
    
    def get_environment_package_path(self):
        raw = subprocess.run(f"{self.env_python} -c 'import site; print(site.getsitepackages())'", capture_output=True, shell=True, text=True).stdout
        trimmed = raw[2:-3]
        return trimmed
    
    def get_environment_packages(self):
        raw = subprocess.run([self.env_pip, "list"], capture_output=True, text=True).stdout
        lines = raw.splitlines()[2:]
        return [line.split()[0] for line in lines]
    
    def validate_node_installed(self, node_module, url):
        node_path = os.path.join(self.nodes_path, node_module)

        if not os.path.exists(node_path): # check all manifest nodes downloaded
            porcelain.clone(url, node_path)
        
        if node_module not in self.env_packages: # check all downloaded nodes installed
            subprocess.check_call([self.env_pip, "install", "-e", node_path])
        
        return os.path.basename(node_path)
    
    def register_module(self, module):
        # Look for an entrypoint named "register"
        pyproject = os.path.abspath(next((f for f in distribution(module).files if "pyproject.toml" in str(f)), None))

        if not os.path.exists(pyproject):
            # No information about which functions to use, so we have to register all of them
            return self.register_all_functions_in_module(module)

        with open(pyproject, 'r') as f:
            data = tomllib.load(f)
        
        tools = data.get("tool", {})

        sdbx = tools.get("sdbx", {})
        comfy = tools.get("comfy", {})

        if not sdbx and not comfy:
            return self.register_all_functions_in_module(module)
        if not sdbx and comfy:
            return self.register_module_with_adapter(module, ComfyAdapter)
        
        register = sdbx.get("register",  None)

        if not register or not callable(register):
            # No register function
            return self.register_all_functions_in_module(module)