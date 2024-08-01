import os
import site
import venv
import tomllib
import subprocess

from dulwich import porcelain

class NodeManager:
    def __init__(self, path, nodes_path, env_name=".node_env"):
        self.path = path
        self.nodes_path = nodes_path
        
        with open(os.path.join(path, "extensions.toml"), 'rb') as file:
            self.nodes = tomllib.load(file)["nodes"]
        
        self.initialize_environment(env_name)
        self.validate_nodes_installed()
    
    def initialize_environment(self, env_name=".node_env"):
        self.env_path = os.path.join(self.path, env_name)

        if not os.path.exists(self.env_path):
            venv.create(self.env_path, with_pip=True)
        
        self.env_python = os.path.join(self.env_path, "bin", "python")
        self.env_package_path = self.get_environment_package_path()

        parent_package_path = site.getsitepackages()[0]
        parent_link_path = os.path.join(self.env_package_path, "parent.pth")
        if not os.path.exists(parent_link_path):
            with open(parent_link_path, 'w') as f:
                f.write(parent_package_path)
        
        child_link_path = os.path.join(parent_package_path, "nodes.pth")
        if not os.path.exists(child_link_path):
            with open(child_link_path, 'w') as f:
                f.write(self.env_package_path)

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
    
    def validate_nodes_installed(self):
        for node, url in self.nodes.items():
            nodepath = os.path.join(self.nodes_path, node)

            if not os.path.exists(nodepath): # are all listed nodes downloaded
                porcelain.clone(url, nodepath)
            
            if node not in self.env_packages: # are all downloaded models installed
                subprocess.check_call([self.env_pip, "install", "-e", nodepath])