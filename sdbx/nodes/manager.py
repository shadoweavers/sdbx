import os
import tomllib

from dulwich import porcelain

class NodeManager:
    def __init__(self, path, nodes_path):
        self.path = path
        self.nodes_path = nodes_path
        
        with open(os.path.join(path, "extensions.toml"), 'rb') as file:
            self.nodes = tomllib.load(file)["nodes"]
        
        self.initialize_environment()
        self.validate_nodes_installed()
    
    def validate_nodes_installed(self):
        for node, url in self.nodes.items():
            nodepath = os.path.join(self.path, node)

            if not os.path.exists(nodepath): # are all listed nodes downloaded
                porcelain.clone(url, nodepath)
            
            # are all downloaded models installed
            # if not in pip freeze, pip install -e nodepath