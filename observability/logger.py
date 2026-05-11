from datetime import datetime

from core.privacy import sanitize_for_logging


def log_step(step: str, message: str, data: dict | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n[{timestamp}] [{step}] {message}")

    if data:
        for key, value in sanitize_for_logging(data).items():
            print(f"  - {key}: {value}")
