import yaml
from box import Box


def load_config(config_path: str = "config.yaml") -> Box:
    with open(config_path) as f:
        return Box(yaml.safe_load(f))
