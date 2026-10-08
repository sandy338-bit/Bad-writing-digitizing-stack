from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

from .images import load_image
from .readers import ReaderUnavailable


class QueueFullError(RuntimeError):
    pass


class Jobs:
    """One worker bounds model concurrency; pending work is capped separately."""
    def __init__(self, store, digitizer, capacity: int):
        self.store = store
        self.digitizer = digitizer
        self.slots = BoundedSemaphore(capacity)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scrawlguard")

    def submit(self, filename: str, image: bytes, correction_enabled: bool = True) -> str:
        if not self.slots.acquire(blocking=False):
            raise QueueFullError("Processing queue is full; try again shortly")
        try:
            identifier = self.store.create(filename, image, correction_enabled)
            self.executor.submit(self._run, identifier)
            return identifier
        except Exception:
            self.slots.release()
            raise

    def retry(self, identifier: str):
        if not self.slots.acquire(blocking=False):
            raise QueueFullError("Processing queue is full; try again shortly")
        try:
            self.store.retry(identifier)
            self.executor.submit(self._run, identifier)
        except Exception:
            self.slots.release()
            raise

    def _run(self, identifier: str):
        try:
            document = self.store.get(identifier)
            self.store.update_job(identifier, "processing")
            result = self.digitizer.transcribe(load_image(self.store.image(identifier)), document["filename"],
                                               correction=bool(document["correction_enabled"]))
            self.store.update_job(identifier, "completed", result=result)
        except ReaderUnavailable as exc:
            self.store.update_job(identifier, "failed", error=str(exc))
        except Exception as exc:
            self.store.update_job(identifier, "failed", error=f"Recognition failed ({type(exc).__name__}); check reader configuration and retry")
        finally:
            self.slots.release()

    def close(self):
        self.executor.shutdown(wait=True)
