
"""Central configuration for GNOMEdata."""

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_name: str = "GNOMEdata"
    environment: str = "development"
    data_dir: str = "data"
    max_upload_mb: int = 10
    chunk_size: int = 800
    chunk_overlap: int = 120

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        """Build and validate settings from environment variables."""
        env = os.environ if environ is None else environ

        settings = cls(
            environment=env.get(
                "GNOMEDATA_ENV", "development"
            ),
            data_dir=env.get(
                "GNOMEDATA_DATA_DIR", "data"
            ),
            max_upload_mb=int(
                env.get("GNOMEDATA_MAX_UPLOAD_MB", "10")
            ),
            chunk_size=int(
                env.get("GNOMEDATA_CHUNK_SIZE", "800")
            ),
            chunk_overlap=int(
                env.get("GNOMEDATA_CHUNK_OVERLAP", "120")
            ),
        )

        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject invalid configuration before the app runs."""
        if self.environment not in {
            "development", "testing", "production"
        }:
            raise ValueError("Invalid GNOMEDATA_ENV.")

        if not self.data_dir.strip():
            raise ValueError("Data directory cannot be empty.")

        if self.max_upload_mb < 1:
            raise ValueError("Maximum upload size must be positive.")

        if self.chunk_size < 1:
            raise ValueError("Chunk size must be positive.")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "Chunk overlap must be non-negative "
                "and smaller than chunk size."
            )


settings = Settings.from_env()
