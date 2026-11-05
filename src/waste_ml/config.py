"""Runtime configuration shared by training, evaluation, and inference."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parents[2]

CLASSES: tuple[str, ...] = ("organic", "plastic", "paper")
CLASS_IDS: dict[str, int] = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SIZE: tuple[int, int] = (224, 224)
INPUT_SHAPE: tuple[int, int, int] = (224, 224, 3)
DEFAULT_MODEL_VERSION = "1.1.0"
DEFAULT_SEED = 42


class Settings(BaseSettings):
    """Environment-driven service settings."""

    model_config = SettingsConfigDict(
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "WasteWaste ML"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_path: Path = Path("artifacts/waste_classifier.keras")
    model_metadata_path: Path = Path("artifacts/model_metadata.json")

    max_upload_size_mb: int = 5

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def resolve_path(self, path: Path) -> Path:
        """Resolve relative artifact paths against the service root."""
        return path if path.is_absolute() else SERVICE_ROOT / path

    @property
    def resolved_model_path(self) -> Path:
        return self.resolve_path(self.model_path)

    @property
    def resolved_model_metadata_path(self) -> Path:
        return self.resolve_path(self.model_metadata_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()
