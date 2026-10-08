from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import ImageDraw
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .export import render_text
from .images import crop, load_image
from .jobs import Jobs, QueueFullError
from .readers import ReaderUnavailable
from .schemas import (BaselineResult, CorrectionRequest, DocumentList, DocumentRecord,
                      JobAccepted, Transcription)
from pydantic import BaseModel, Field
from .service import Digitizer
from .store import ConflictError, Store

logger = logging.getLogger(__name__)


class PageReviewRequest(BaseModel):
    revision: int = Field(ge=1)


class ManualTranscriptionRequest(BaseModel):
    revision: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=20000)


def png_bytes(image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def create_app(settings: Settings | None = None, digitizer=None) -> FastAPI:
    settings = settings or Settings()
    engine = digitizer or Digitizer(settings)

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(settings.database_path)
        app.state.store.recover()
        app.state.jobs = Jobs(app.state.store, engine, settings.max_pending_jobs)
        yield
        await run_in_threadpool(app.state.jobs.close)

    app = FastAPI(title="ScrawlGuard API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware,
                       allow_origins=[s.strip() for s in settings.cors_origins.split(",") if s.strip()],
                       allow_credentials=False, allow_methods=["GET", "POST", "PATCH", "DELETE"],
                       allow_headers=["*"], expose_headers=["Content-Disposition"])

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse(status_code=404, content={"detail": "Document or region not found"})

    @app.exception_handler(ConflictError)
    async def conflict(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(QueueFullError)
    async def queue_full(request, exc):
        return JSONResponse(status_code=429, content={"detail": str(exc)}, headers={"Retry-After": "5"})

    async def image_upload(file: UploadFile):
        try:
            if file.content_type not in {"image/png", "image/jpeg", "image/webp", "image/tiff"}:
                raise HTTPException(415, "Upload PNG, JPEG, WebP, or single-page TIFF")
            raw = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(raw) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, "Image exceeds upload limit")
            try:
                return await run_in_threadpool(load_image, raw)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()

    def completed(identifier: str):
        document = app.state.store.get(identifier)
        if document["status"] != "completed":
            raise ConflictError("Document processing has not completed")
        return document, Transcription.model_validate(document["result"])

    async def run_model(method, image, filename):
        if not app.state.jobs.slots.acquire(blocking=False):
            raise QueueFullError("Processing queue is full; try again shortly")
        try:
            return await run_in_threadpool(method, image, filename)
        finally:
            app.state.jobs.slots.release()

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "1.0.0", "capabilities": engine.capabilities()}

    @app.get("/", include_in_schema=False)
    def frontend():
        return FileResponse(Path(__file__).resolve().parent.parent / "frontend" / "index.html",
                            media_type="text/html")

    @app.post("/v1/transcribe", response_model=Transcription)
    async def transcribe(file: UploadFile = File(...)):
        image = await image_upload(file)
        try:
            return await run_model(engine.transcribe, image, file.filename or "image")
        except QueueFullError:
            raise
        except ReaderUnavailable as exc:
            raise HTTPException(429 if exc.reason == "quota" else 503, str(exc)) from exc
        except Exception as exc:
            logger.exception("Transcription unavailable")
            raise HTTPException(503, "Recognition unavailable; check reader configuration") from exc

    @app.post("/v1/baseline", response_model=BaselineResult)
    async def baseline(file: UploadFile = File(...)):
        image = await image_upload(file)
        try:
            return await run_model(engine.baseline, image, file.filename or "image")
        except QueueFullError:
            raise
        except ReaderUnavailable as exc:
            raise HTTPException(429 if exc.reason == "quota" else 503, str(exc)) from exc
        except Exception as exc:
            logger.exception("Baseline unavailable")
            raise HTTPException(503, "Baseline model unavailable") from exc

    @app.post("/v1/documents", status_code=202, response_model=JobAccepted)
    async def submit(file: UploadFile = File(...), correction: bool = True):
        image = await image_upload(file)
        normalized = await run_in_threadpool(png_bytes, image)
        identifier = await run_in_threadpool(app.state.jobs.submit, (file.filename or "image")[:255], normalized, correction)
        return {"id": identifier, "status": "queued", "status_url": f"/v1/documents/{identifier}"}

    @app.get("/v1/documents", response_model=DocumentList)
    def documents(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
        return app.state.store.list(limit, offset)

    @app.get("/v1/documents/{identifier}", response_model=DocumentRecord)
    def document(identifier: str):
        return app.state.store.get(identifier)

    @app.post("/v1/documents/{identifier}/retry", status_code=202, response_model=JobAccepted)
    def retry(identifier: str):
        app.state.store.get(identifier)
        app.state.jobs.retry(identifier)
        return {"id": identifier, "status": "queued"}

    @app.post("/v1/documents/{identifier}/manual-transcription", response_model=DocumentRecord)
    def manual_transcription(identifier: str, request: ManualTranscriptionRequest):
        if not request.text.strip():
            raise HTTPException(422, "Enter visible text or [[illegible]]")
        return app.state.store.complete_manual(identifier, request.revision, request.text.strip())

    @app.patch("/v1/documents/{identifier}/corrections", response_model=DocumentRecord)
    def corrections(identifier: str, request: CorrectionRequest):
        try:
            return app.state.store.correct(identifier, request)
        except ConflictError:
            raise
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/documents/{identifier}/history")
    def history(identifier: str):
        return {"items": app.state.store.history(identifier)}

    @app.post("/v1/documents/{identifier}/review-page", response_model=DocumentRecord)
    def review_page(identifier: str, request: PageReviewRequest):
        return app.state.store.review_page(identifier, request.revision)

    @app.get("/v1/documents/{identifier}/review-queue")
    def review_queue(identifier: str,
                     min_agreement: int = Query(2, ge=1, le=10),
                     strict_high_risk: bool = True, medical_terms: bool = False):
        _, result = completed(identifier)
        words = []
        for region in result.regions:
            for word in region.words:
                if word.eligibility in {"verified", "excluded", "illegible"}:
                    continue
                flagged = (word.eligibility == "needs_review" or word.readers_total < 2 or
                           word.readers_agree < min_agreement or
                           word.confidence < settings.accept_threshold)
                if strict_high_risk and word.risk_tag != "normal":
                    flagged = flagged or word.readers_total < 2 or word.readers_agree < word.readers_total
                if medical_terms and word.risk_tag == "medical_term":
                    flagged = True
                if flagged:
                    value = word.model_copy(deep=True)
                    if value.eligibility != "illegible":
                        value.eligibility = "needs_review"
                        value.status = "review"
                    words.append(value)
        words.sort(key=lambda word: (word.risk, word.risk_tag != "normal"), reverse=True)
        return {"total": len(words), "min_agreement": min_agreement,
                "strict_high_risk": strict_high_risk,
                "medical_terms": medical_terms,
                "items": [word.model_dump() for word in words]}

    @app.get("/v1/documents/{identifier}/words/{word_id}/image")
    def word_image(identifier: str, word_id: str):
        _, result = completed(identifier)
        word = next((word for region in result.regions for word in region.words if word.id == word_id), None)
        if word is None:
            raise KeyError(word_id)
        source = load_image(app.state.store.image(identifier))
        if word.source_scope == "page":
            return Response(png_bytes(source), media_type="image/png")
        return Response(png_bytes(crop(source, word.box, padding=5)), media_type="image/png")

    @app.get("/v1/documents/{identifier}/image")
    def image(identifier: str, annotated: bool = False):
        raw = app.state.store.image(identifier)
        if annotated:
            _, result = completed(identifier)
            rendered = load_image(raw)
            draw = ImageDraw.Draw(rendered)
            colors = {"accepted": "green", "verified": "blue", "review": "orange",
                      "illegible": "red", "crossed_out": "purple"}
            for region in result.regions:
                b = region.box
                draw.rectangle((b.x1, b.y1, b.x2, b.y2), outline=colors[region.status], width=3)
                draw.text((b.x1, max(0, b.y1 - 12)), region.id, fill=colors[region.status])
            raw = png_bytes(rendered)
        return Response(raw, media_type="image/png")

    @app.get("/v1/documents/{identifier}/regions/{region_id}/image")
    def region_image(identifier: str, region_id: str):
        _, result = completed(identifier)
        region = next((r for r in result.regions if r.id == region_id), None)
        if region is None:
            raise KeyError(region_id)
        return Response(png_bytes(crop(load_image(app.state.store.image(identifier)), region.box)), media_type="image/png")

    @app.get("/v1/documents/{identifier}/export")
    def export(identifier: str, format: Literal["txt", "md", "json"] = "txt"):
        document, result = completed(identifier)
        headers = {"Content-Disposition": f'attachment; filename="scrawlguard-{identifier}.{format}"'}
        if format == "json":
            document["history"] = app.state.store.history(identifier)
            return JSONResponse(document, headers=headers)
        text = render_text(result)
        if format == "md":
            text = "```text\n" + text.replace("```", "` ` `") + "\n```\n"
        return Response(text, media_type="text/plain" if format == "txt" else "text/markdown", headers=headers)

    @app.delete("/v1/documents/{identifier}", status_code=204)
    def delete(identifier: str):
        app.state.store.delete(identifier)
        return Response(status_code=204)

    return app


app = create_app()
