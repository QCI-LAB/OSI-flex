from pathlib import Path
from pprint import pprint
import yaml
from src.config import OSIFlexConfig


def load_config(path: str) -> OSIFlexConfig:
    config_path = Path(path).resolve()
    project_root = config_path.parent.parent  # go 2 levels up from configs/

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Fix paths for input images
    config_dict["input_images"] = [
        str((project_root / img_path).resolve()) for img_path in config_dict["input_images"]
    ]

    return OSIFlexConfig(**config_dict)


def print_config(config_path: str | Path) -> None:
    """
    Load a YAML configuration file and print its settings.

    Args:
        config_path (str | Path): Path to the YAML config file.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("\n=== Configuration Settings ===\n")
    pprint(config, sort_dicts=False)  # preserves key order from YAML
    print("\n")

