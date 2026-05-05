import json
import os

class Config:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.data = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            # Try looking one level up if not found (for scripts in subdirectories)
            alt_path = os.path.join("..", self.config_path)
            if os.path.exists(alt_path):
                self.config_path = alt_path
            else:
                raise FileNotFoundError(f"Configuration file {self.config_path} not found.")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get(self, *keys, default=None):
        curr = self.data
        try:
            for key in keys:
                curr = curr[key]
            return curr
        except (KeyError, TypeError):
            return default

# Singleton instance for easy access
config = None
try:
    config = Config()
except Exception:
    # Handle cases where config might not be available yet
    pass
