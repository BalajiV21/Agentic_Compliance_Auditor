"""
Configuration management for Agentic Compliance Auditor
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # LLM Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3.2")

    # Vector Database
    CHROMA_PERSIST_DIR: str = Field(default="./data/chroma_db")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")

    # Memory Systems
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)

    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="password")

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=4)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="./logs/app.log")

    # Retrieval Configuration
    TOP_K_RESULTS: int = Field(default=5)
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=50)
    RERANK_ENABLED: bool = Field(default=False)

    # Agent Configuration
    MAX_ITERATIONS: int = Field(default=10)
    AGENT_TIMEOUT: int = Field(default=300)
    ENABLE_SELF_REFLECTION: bool = Field(default=True)

    # Evaluation
    EVAL_DATASET_PATH: str = Field(default="./data/eval_dataset.json")
    RAGAS_METRICS: str = Field(default="faithfulness,answer_relevancy,context_precision,context_recall")

    class Config:
        env_file = ".env"
        case_sensitive = True

    def get_ragas_metrics(self) -> List[str]:
        """Parse RAGAS metrics from comma-separated string"""
        return [m.strip() for m in self.RAGAS_METRICS.split(",")]

    def ensure_dirs(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.DATA_DIR,
            self.DATA_DIR / "raw",
            self.DATA_DIR / "processed",
            self.DATA_DIR / "sample_docs",
            self.LOGS_DIR,
            Path(self.CHROMA_PERSIST_DIR),
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_dirs()
