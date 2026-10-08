"""Explicit, opt-in selected-reader smoke test. Uses only a generated public test image."""
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tempfile
import time

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from pipeline.api import create_app
from pipeline.config import Settings


def main():
    image = Image.new("RGB", (400, 120), "white")
    ImageDraw.Draw(image).text((30, 30), "Hello", font=ImageFont.load_default(size=40), fill="black")
    output = BytesIO()
    image.save(output, format="PNG")
    with tempfile.TemporaryDirectory() as directory:
        settings = replace(Settings(), enable_paddle=False,
                           database_path=str(Path(directory) / "smoke.sqlite3"))
        with TestClient(create_app(settings)) as client:
            response = client.post("/v1/documents", files={"file": ("synthetic.png", output.getvalue(), "image/png")})
            response.raise_for_status()
            identifier = response.json()["id"]
            for _ in range(150):
                document = client.get(f"/v1/documents/{identifier}").json()
                if document["status"] in {"completed", "failed"}:
                    break
                time.sleep(1)
            if document["status"] != "completed":
                raise SystemExit(f"Live smoke failed: {document['status']} {document.get('error')}")
            result = document["result"]
            assert result["regions"] and result["page_status"] == "review"
            assert all(r["status"] != "accepted" for r in result["regions"])
            export = client.get(f"/v1/documents/{identifier}/export")
            export.raise_for_status()
            assert "[[page requires verification]]" in export.text
            print({"status": document["status"], "regions": len(result["regions"]),
                   "page_status": result["page_status"], "text": result["text"], "warnings": result["warnings"]})


if __name__ == "__main__":
    main()
