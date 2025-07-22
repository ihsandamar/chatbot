import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    A simple config wrapper that allows attribute access,
    and exposes llm.api_key also at the top level via instance.api_key
    """
    def __init__(self, data: dict):
        for key, val in data.items():
            if isinstance(val, dict):
                setattr(self, key, Config(val))
            else:
                setattr(self, key, val)


class ConfigLoader:
    @staticmethod
    def load_config(path: str = "config/config.yaml") -> Config:
        # Load .env from config folder if exists
        dotenv_path = Path("config/.env")
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def resolve_env_vars(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_key = value[2:-1]
                return os.getenv(env_key)
            elif isinstance(value, dict):
                return {k: resolve_env_vars(v) for k, v in value.items()}
            return value

        resolved = {k: resolve_env_vars(v) for k, v in raw.items()}
        return Config(resolved)
