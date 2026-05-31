"""FastAPI request / response schemas."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SynthRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Input text (Arabic/English/mixed)")
    language: str = Field("ar", description="'ar' or 'en'")
    speaker_wav: Optional[str] = Field(None, description="Path or URL to reference speaker WAV")
    stream: bool = Field(True, description="Return streaming audio chunks")
    normalize: bool = Field(True, description="Run text normalization pipeline")
    format: str = Field("pcm", description="Output format: 'pcm' (raw s16le) or 'wav'")
    chunk_size: int = Field(20, ge=1, le=200, description="XTTS stream chunk size (tokens)")

    @field_validator("language")
    @classmethod
    def validate_lang(cls, v):
        if v not in ("ar", "en"):
            raise ValueError("language must be 'ar' or 'en'")
        return v

    @field_validator("format")
    @classmethod
    def validate_fmt(cls, v):
        if v not in ("pcm", "wav"):
            raise ValueError("format must be 'pcm' or 'wav'")
        return v


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    vram_allocated_gb: float
    sample_rate: int


class MetricsResponse(BaseModel):
    requests_total: int
    ttfa_p50_ms: float
    ttfa_p95_ms: float
    rtf_p50: float
    rtf_p95: float
    peak_vram_gb: float
