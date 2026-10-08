from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Box(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_bounds(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")
        return self


class Candidate(BaseModel):
    source: str
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class WordResult(BaseModel):
    id: str
    region_id: str
    line_index: int = Field(ge=0)
    position: int = Field(ge=0)
    box: Box
    source_scope: Literal["word_estimate", "line_estimate", "page"] = "word_estimate"
    text: str
    status: Literal["accepted", "review", "illegible", "crossed_out", "verified"]
    eligibility: Literal["eligible", "needs_review", "illegible", "excluded", "verified"]
    legibility: Literal["legible", "uncertain", "illegible", "struck"]
    confidence: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    risk_tag: Literal["normal", "number", "unit", "negation", "medical_term", "name"] = "normal"
    candidates: list[Candidate] = Field(default_factory=list)
    readers_agree: int = Field(ge=0)
    readers_total: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)


class Region(BaseModel):
    id: str
    box: Box
    kind: Literal["body", "margin", "crossed_out", "unknown"] = "unknown"
    candidates: list[Candidate] = Field(default_factory=list)
    text: str
    status: Literal["accepted", "review", "illegible", "crossed_out", "verified"]
    confidence: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    words: list[WordResult] = Field(default_factory=list)


class Transcription(BaseModel):
    filename: str
    width: int
    height: int
    text: str
    language: str = "Unknown"
    page_status: Literal["accepted", "review"] = "review"
    page_candidate: str | None = None
    page_candidates: list[Candidate] = Field(default_factory=list)
    regions: list[Region]
    warnings: list[str] = Field(default_factory=list)
    model_info: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, int | float] = Field(default_factory=dict)


class BaselineResult(BaseModel):
    filename: str
    text: str
    model: str


class RegionEdit(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(max_length=20000)


class WordEdit(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    text: str = Field(default="", max_length=500)
    action: Literal["replace", "accept", "illegible"] = "replace"


class CorrectionRequest(BaseModel):
    revision: int = Field(ge=1)
    edits: list[RegionEdit] = Field(default_factory=list, max_length=200)
    word_edits: list[WordEdit] = Field(default_factory=list, max_length=500)
    page_reviewed: bool = False

    @model_validator(mode="after")
    def has_edits(self):
        if not self.edits and not self.word_edits:
            raise ValueError("Provide at least one region or word edit")
        return self


class JobAccepted(BaseModel):
    id: str
    status: Literal["queued"]
    status_url: str | None = None


class DocumentSummary(BaseModel):
    id: str
    filename: str
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: str
    updated_at: str
    revision: int
    error: str | None = None
    correction_enabled: bool = True


class DocumentRecord(DocumentSummary):
    result: Transcription | None = None


class DocumentList(BaseModel):
    total: int
    items: list[DocumentSummary]
