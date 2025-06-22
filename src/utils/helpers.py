import yaml


def load_config(path: str):
    """Load YAML config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
