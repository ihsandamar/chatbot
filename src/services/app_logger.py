# src/services/app_logger.py
"""
Görsel (Rich) + yapısal (structlog) + ortam‑duyarlı merkezi logger.

Kullanım:
    from src.services.app_logger import log  # tek satır!!!
    log.info("başladı", node="chatbot")

Tasarım kararları
-----------------
1. **Kök (root) logger** WARNING seviyesinde tutulur → 3rd‑party kütüphane INFO
   mesajları konsola düşmez.
2. **Uygulama loggerʼı** (adı `dynamic-reporting`) envʼe göre DEBUG/INFOʼda
   çalışır; renkli Rich çıktısı veya prodʼda JSON sağlar.
3. İsteğe bağlı dosya çıktısı (`LOG_TO_FILE=1`) "logs/app.log"a yazılır.
4. Tek seferlik lazy yapı: `log = _LazyLogger()` — ilk kullanımda konfigüre olur.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import structlog
from rich.logging import RichHandler

__all__ = ["log"]

# ─────────────────────────── Ortam Tanımı ────────────────────────────── #

class Env(str, Enum):
    PROD = "prod"
    DEV = "dev"
    DEBUG = "debug"


# ───────────────────────────── Setup Fonksiyonu ───────────────────────── #

def _init_logger(
    env: Optional[str] = None,
    *,
    log_to_file: bool = False,
    log_dir: str | Path = "logs",
) -> None:
    """Merkezî loggerʼı **bir kez** konfigüre eder.

    Sonraki `structlog.get_logger()` çağrılarında cache kullanılır.
    """

    env = Env(env or os.getenv("APP_ENV", "dev").lower())
    app_level = logging.DEBUG if env in (Env.DEV, Env.DEBUG) else logging.INFO

    # 1) KÖK LOGGER → WARNING
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # 2) Handler kurulumu (Rich + opsiyonel File)
    handlers: list[logging.Handler] = [
        RichHandler(markup=True, rich_tracebacks=True)
    ]
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(Path(log_dir, "app.log"), encoding="utf-8")
        )

    # Handlers'ı kök loggere ekle (basicConfig sadece ilk çağrıda etkili)
    logging.basicConfig(format="%(message)s", handlers=handlers)

    # 3) UYGULAMA LOGGER'ı adı "dynamic-reporting"
    app_logger = logging.getLogger("dynamic-reporting")
    app_logger.setLevel(app_level)

    # 4) Gürültülü kütüphaneleri sustur (opsiyonel)
    for noisy in ["langchain", "sqlalchemy", "httpx", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 5) structlog konfigürasyonu
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if env is Env.PROD:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(app_level),
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ───────────────────────────── Kullanıcı APIʼsi ────────────────────────── #

class _LazyLogger:
    """İlk erişimde `_init_logger`ʼı çağıran proxy."""

    _configured = False

    def __call__(self, *args, **kwargs):  # log(...)
        return self.get()(*args, **kwargs)

    def get(self, **bind):
        if not self._configured:
            _init_logger(
                log_to_file=os.getenv("LOG_TO_FILE", "0") == "1",
                log_dir=os.getenv("LOG_DIR", "logs"),
            )
            self._configured = True
        # Uygulama loggerʼı adı → "dynamic-reporting"
        return structlog.get_logger("dynamic-reporting").bind(**bind)


# Tek satırlık global erişim noktası
log = _LazyLogger()


# ───────────────────────────── Demo Modu ──────────────────────────────── #

if __name__ == "__main__":
    demo = log.get(module="demo")
    demo.debug("demo DEBUG log")
    try:
        1 / 0
    except ZeroDivisionError:
        demo.exception("sıfıra bölme denendi", user_id=42)
