"""SQLite document storage. Images use database blobs, never client filenames as paths."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .images import load_image
from .schemas import Box, Candidate, CorrectionRequest, Region, Transcription
from .words import analyze_region, populate_words, summarize_words


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def refresh_legacy_scores(result: dict | None) -> dict | None:
    if not result:
        return result
    for region in result.get("regions", []):
        for word in region.get("words", []):
            if word.get("eligibility") not in {"needs_review", "illegible"} or len(word.get("candidates", [])) >= 2:
                continue
            word["confidence"] = min(word.get("confidence", 0), 0.35)
            priority = 1.0 if word["eligibility"] == "illegible" else min(
                1.0, 0.65 * (1.25 if word.get("risk_tag") != "normal" else 1.0))
            word["risk"] = max(word.get("risk", 0), round(priority, 3))
    return result


class ConflictError(ValueError):
    pass


class Store:
    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, image BLOB NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1, result TEXT, error TEXT,
                correction_enabled INTEGER NOT NULL DEFAULT 1)""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            if "correction_enabled" not in columns:
                db.execute("ALTER TABLE documents ADD COLUMN correction_enabled INTEGER NOT NULL DEFAULT 1")
            db.execute("""CREATE TABLE IF NOT EXISTS corrections (
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                revision INTEGER NOT NULL, created_at TEXT NOT NULL, changes TEXT NOT NULL)""")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover(self):
        with self.connection() as db:
            db.execute("UPDATE documents SET status='failed', error=?, updated_at=? WHERE status IN ('queued','processing')",
                       ("Processing was interrupted by a server restart; retry this document", now()))

    def create(self, filename: str, image: bytes, correction_enabled: bool = True) -> str:
        identifier = uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO documents(id,filename,image,status,created_at,updated_at,correction_enabled) VALUES(?,?,?,?,?,?,?)",
                       (identifier, filename, image, "queued", now(), now(), correction_enabled))
        return identifier

    def get(self, identifier: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT id,filename,status,created_at,updated_at,revision,result,error,correction_enabled FROM documents WHERE id=?",
                             (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        result = dict(row)
        result["result"] = refresh_legacy_scores(json.loads(result["result"])) if result["result"] else None
        return result

    def image(self, identifier: str) -> bytes:
        with self.connection() as db:
            row = db.execute("SELECT image FROM documents WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return row["image"]

    def list(self, limit: int, offset: int) -> dict:
        with self.connection() as db:
            total = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            rows = db.execute("SELECT id,filename,status,created_at,updated_at,revision,error,correction_enabled FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                              (limit, offset)).fetchall()
        return {"total": total, "items": [dict(row) for row in rows]}

    def update_job(self, identifier: str, status: str, result: Transcription | None = None, error: str | None = None):
        with self.connection() as db:
            db.execute("UPDATE documents SET status=?, result=?, error=?, updated_at=? WHERE id=?",
                       (status, result.model_dump_json() if result else None, error, now(), identifier))

    def retry(self, identifier: str):
        with self.connection() as db:
            cursor = db.execute("UPDATE documents SET status='queued', error=NULL, updated_at=? WHERE id=? AND status='failed'", (now(), identifier))
            if not cursor.rowcount:
                raise ConflictError("Only failed documents can be retried")

    def complete_manual(self, identifier: str, revision: int, text: str) -> dict:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,revision,image,filename FROM documents WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            if row["status"] != "failed" or row["revision"] != revision:
                raise ConflictError("Manual transcription needs a failed document and its latest revision")
            image = load_image(row["image"])
            region = Region(id="r1", box=Box(x1=0, y1=0, x2=image.width, y2=image.height),
                            text=text, status="review", confidence=0,
                            candidates=[Candidate(source="human_entry", text=text)],
                            reasons=["Entered by a person after automatic recognition was unavailable",
                                     "Source location is the whole page"])
            result = Transcription(filename=row["filename"], width=image.width, height=image.height,
                                   text=text, page_status="review", regions=[region],
                                   warnings=["Manual transcription; individual word locations are not verified"],
                                   model_info={"ocr": "unavailable", "vision": "unavailable"})
            result.summary = populate_words(result.regions)
            for word in region.words:
                word.source_scope = "page"
            next_revision = revision + 1
            db.execute("UPDATE documents SET status='completed',result=?,error=NULL,revision=?,updated_at=? WHERE id=?",
                       (result.model_dump_json(), next_revision, now(), identifier))
            db.execute("INSERT INTO corrections VALUES(?,?,?,?)",
                       (identifier, next_revision, now(), json.dumps({"manual_transcription": True})))
        return self.get(identifier)

    def correct(self, identifier: str, request: CorrectionRequest) -> dict:
        if len({edit.id for edit in request.edits}) != len(request.edits) or len({edit.id for edit in request.word_edits}) != len(request.word_edits):
            raise ValueError("Edit IDs must be unique")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM documents WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            if row["status"] != "completed" or row["revision"] != request.revision:
                raise ConflictError("Document is not completed or revision is stale; reload it")
            result = Transcription.model_validate_json(row["result"])
            by_id = {region.id: region for region in result.regions}
            if any(edit.id not in by_id for edit in request.edits):
                raise ValueError("Unknown region ID")
            changes = []
            for edit in request.edits:
                region = by_id[edit.id]
                changes.append({"id": edit.id, "before": region.text, "after": edit.text,
                                "previous_status": region.status})
                region.text = edit.text
                region.status = "verified"
                region.reasons = ["Verified by a human reviewer"]
                region.words = analyze_region(region, next(i for i, item in enumerate(result.regions) if item.id == region.id))
            words = {word.id: (region, word) for region in result.regions for word in region.words}
            if any(edit.id not in words for edit in request.word_edits):
                raise ValueError("Unknown word ID")
            for edit in request.word_edits:
                region, word = words[edit.id]
                before = word.text
                replacement = "[[illegible]]" if edit.action == "illegible" else (edit.text or word.text)
                changes.append({"id": edit.id, "before": before, "after": replacement,
                                "previous_status": word.status, "level": "word"})
                word.text = replacement
                word.status = "illegible" if edit.action == "illegible" else "verified"
                word.eligibility = "illegible" if edit.action == "illegible" else "verified"
                word.legibility = "illegible" if edit.action == "illegible" else "legible"
                word.reasons = ["Marked illegible by a human reviewer"] if edit.action == "illegible" else ["Verified by a human reviewer"]
                region.text = " ".join(item.text for item in region.words)
                region.status = "illegible" if any(item.status == "illegible" for item in region.words) else (
                    "verified" if all(item.status in {"accepted", "verified", "crossed_out"} for item in region.words) else "review")
            result.text = "\n".join("[crossed out]" if r.status == "crossed_out" else r.text for r in result.regions)
            result.summary = summarize_words(result.regions)
            if request.page_reviewed and all(r.status in {"accepted", "verified", "crossed_out"} for r in result.regions):
                result.page_status = "accepted"
            revision = row["revision"] + 1
            db.execute("UPDATE documents SET result=?, revision=?, updated_at=? WHERE id=?",
                       (result.model_dump_json(), revision, now(), identifier))
            db.execute("INSERT INTO corrections VALUES(?,?,?,?)", (identifier, revision, now(),
                       json.dumps({"edits": changes, "page_reviewed": request.page_reviewed}, ensure_ascii=False)))
        return self.get(identifier)

    def history(self, identifier: str) -> list[dict]:
        self.get(identifier)
        with self.connection() as db:
            rows = db.execute("SELECT revision,created_at,changes FROM corrections WHERE document_id=? ORDER BY revision", (identifier,)).fetchall()
        return [{"revision": r["revision"], "created_at": r["created_at"], "changes": json.loads(r["changes"])} for r in rows]

    def review_page(self, identifier: str, revision: int) -> dict:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,revision,result FROM documents WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            if row["status"] != "completed" or row["revision"] != revision:
                raise ConflictError("Document is not completed or revision is stale; reload it")
            result = Transcription.model_validate_json(row["result"])
            if any(word.eligibility == "needs_review"
                   for region in result.regions for word in region.words):
                raise ConflictError("Review all uncertain words before verifying the page")
            result.page_status = "accepted"
            next_revision = revision + 1
            db.execute("UPDATE documents SET result=?, revision=?, updated_at=? WHERE id=?",
                       (result.model_dump_json(), next_revision, now(), identifier))
            db.execute("INSERT INTO corrections VALUES(?,?,?,?)",
                       (identifier, next_revision, now(), json.dumps({"edits": [], "page_reviewed": True})))
        return self.get(identifier)

    def delete(self, identifier: str):
        with self.connection() as db:
            row = db.execute("SELECT status FROM documents WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            if row["status"] in {"queued", "processing"}:
                raise ConflictError("Wait for processing to finish before deleting")
            db.execute("DELETE FROM documents WHERE id=?", (identifier,))
