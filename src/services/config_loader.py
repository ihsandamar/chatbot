# src/services/config_loader.py
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

class ConfigLoader:
    @staticmethod
    def load_config(path: str = "src/config/config.yaml") -> dict:
        # .env dosyasını config klasörü içinde ara
        dotenv_path = Path("src/config/.env")
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Ortam değişkenlerini çözümle
        def resolve_env_vars(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_key = value[2:-1]
                return os.getenv(env_key)
            elif isinstance(value, dict):
                return {k: resolve_env_vars(v) for k, v in value.items()}
            return value

        return {k: resolve_env_vars(v) for k, v in raw.items()}
