import logging

import yaml
from box import Box


def load_config(config_path: str = "config.yaml") -> Box:
    with open(config_path) as f:
        return Box(yaml.safe_load(f))


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent timestamped format.

    Call once at the start of train.py. All module-level loggers created with
    ``logging.getLogger(__name__)`` will inherit this configuration.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
